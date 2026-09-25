import numpy as np
from scipy import sparse, linalg
import time
from joblib import Parallel, delayed, parallel_config

from . import utils


def build_smoothness_operator(galaxy_list):
    """Build a block-diagonal discrete second-derivative operator D.

    D has one block per galaxy, in the same order as `galaxy_list`. Each
    block acts on that galaxy's SED coefficients (its slice of the flat
    parameter vector `x`, as produced by `unpack_extractions`) and
    computes the discrete second derivative -- the local curvature --
    of the spectrum along the wavelength grid:

        (D_i @ x_i)[r] = x_i[r] - 2*x_i[r+1] + x_i[r+2]

    Penalizing ||D @ x||^2 (instead of ||x||^2, as plain ridge does)
    therefore penalizes wiggliness/roughness in each galaxy's spectrum
    rather than penalizing its overall flux level: a constant offset or
    a linear trend has zero second derivative, so those are left
    unpenalized while noisy high-frequency structure is suppressed.

    The blocks are stacked block-diagonally (galaxy 0's rows/columns,
    then galaxy 1's, etc.) so that D never compares coefficients that
    belong to two different galaxies -- this mirrors how
    `compute_jacobian` builds the Jacobian by horizontally stacking one
    block of columns per galaxy, and how `unpack_extractions` later
    splits `x` back into per-galaxy chunks in that same order.

    Parameters
    ----------
    galaxy_list : list
        Galaxies in the same order used to build the Jacobian / `x`
        (each contributes `len(galaxy.wavelength)` columns to D).

    Returns
    -------
    D : scipy.sparse csr array, shape (n_rows, n_cols)
        `n_cols` equals `len(x)` (sum of `len(gal.wavelength)` over all
        galaxies). `n_rows` is the sum over galaxies of
        `max(len(gal.wavelength) - 2, 0)` -- galaxies with fewer than 3
        wavelength samples have no interior point to take a second
        difference of, so they contribute zero rows (no smoothness
        constraint on them, but they still need their columns present
        so the block widths add up to `len(x)`).
    """
    blocks = []
    for gal in galaxy_list:
        n = len(gal.wavelength)

        if n < 3:
            # Too few samples for a second difference: an all-zero
            # block with the right number of columns and no rows.
            blocks.append(sparse.csr_array((0, n)))
            continue

        n_rows = n - 2

        # Row r reads off columns r, r+1, r+2 (the 3-point stencil for
        # the second derivative), with coefficients 1, -2, 1.
        # row_idx: [0,0,0, 1,1,1, 2,2,2, ...]   (each row index repeated 3x)
        # col_idx: [0,1,2, 1,2,3, 2,3,4, ...]   (each row's own 3 columns)
        # data:    [1,-2,1, 1,-2,1, 1,-2,1, ...]
        row_idx = np.repeat(np.arange(n_rows), 3)
        col_idx = (np.arange(n_rows)[:, None] + np.arange(3)[None, :]).ravel()
        data = np.tile([1., -2., 1.], n_rows)

        D_i = sparse.coo_array((data, (row_idx, col_idx)), shape=(n_rows, n)).tocsr()
        blocks.append(D_i)

    # Stack all per-galaxy blocks diagonally into one sparse operator
    # spanning the full parameter vector.
    D = sparse.block_diag(blocks, format='csr')
    return D


def pseudoinv(Q, r=1e-15, k=None):
    """ """
    if k is None:
        k = Q.shape[0]-1

    if k >= Q.shape[0]:
        k = Q.shape[0]-1

    u, s, v = sparse.linalg.svds(Q, k=k)

    sinv = np.zeros(len(s))

    threshold = s.max()*r

    nonzero = s > threshold
    print(f"number of singular values {np.sum(nonzero)} of {Q.shape[0]}")

    sinv[nonzero] = 1./s[nonzero]

    Qinv = np.dot(v.T*sinv, u.T)

    return Qinv


def trace_product(Qinv, M):
    """Compute trace(Qinv @ M) without ever forming the n x n product.

    trace(Qinv @ M) = sum_ij Qinv_ij * M_ji, so the matmul -- O(n^3)
    flops plus a second dense n x n allocation -- is unnecessary. An
    elementwise product summed is O(n^2) dense, or O(nnz) when M is
    sparse. With n_params in the thousands that is the difference
    between seconds and milliseconds, and it is paid once per lambda
    per masking iteration, so it is worth the two lines.

    Both matrices this is called with (Q_unreg, and the target-row
    Gram matrix) are symmetric Gram matrices, so the transpose is a
    formality -- but it costs nothing (`.T` on a csr array is an O(1)
    view) and keeps the identity exact for any M.
    """
    if sparse.issparse(M):
        return float(M.T.multiply(Qinv).sum())
    return float(np.sum(Qinv * np.asarray(M).T))


def solve(A, b, b_var, lam=1e-3, compute_var=False,
          reg_operator=None, reg_floor=1e-3, dof_rows=None):
    """Weighted least-squares solve of A @ x ~= b, with Tikhonov regularization.

    Solves the regularized normal equations

        (A^T Cinv A + lam_eff * P) x = A^T Cinv b

    where Cinv = diag(1/b_var) and P is the regularization matrix:
      - plain ridge (shrink-to-zero), if `reg_operator` is None: P = I,
        so the penalty term is lam_eff * ||x||^2.
      - smoothness (curvature) penalty, if `reg_operator` (e.g. from
        `build_smoothness_operator`) is given: P = reg_operator.T @
        reg_operator, so the penalty term is lam_eff * ||reg_operator @
        x||^2 -- a small ridge floor is added on top, see `reg_floor`.

    Parameters
    ----------
    A : sparse array
        Design matrix (Jacobian), rows = data points, columns = model
        parameters.
    b : ndarray
        Data vector.
    b_var : ndarray
        Per-data-point variance.
    lam : float, optional
        Regularization strength; the matrix is scaled by the typical
        size of `A`'s entries (see `lam_eff` below) so that a given
        `lam` has a comparable effect regardless of the overall
        flux/units scale of `A`. `lam <= 0` disables regularization.
    compute_var : bool, optional
        If True, also return (an approximation of) the covariance of
        `x`, in the old post-hoc-smoothing style. See `smooth` above.
    reg_operator : sparse array, optional
        Linear operator D such that ||D @ x||^2 is the penalty to
        minimize, instead of the plain ridge penalty ||x||^2.
    reg_floor : float, optional
        Only used when `reg_operator` is given. Fraction (relative to
        the plain ridge strength) of a plain ||x||^2 penalty that is
        always added on top of the smoothness penalty. A pure
        second-difference penalty has a null space -- a constant
        offset or a linear trend has zero curvature -- so for a galaxy
        that's only weakly constrained by data, the normal equations
        can come out singular without this floor to fall back on.
    dof_rows : ndarray of bool, optional
        Boolean mask over the rows of `A` (i.e. over the entries of
        `b`) selecting a subset of the data. When given, the effective
        degrees of freedom are *also* computed restricted to those
        rows, and returned as `dof_eff_rows`. This is the diagonal sum
        of the hat matrix over that subset only -- how much freedom the
        fit spends on those particular pixels -- which is what pairs
        with a chi2 computed on the same subset to form a valid
        GCV-style score. The full parameter vector is still fit to all
        rows; only the accounting is restricted.

    Returns
    -------
    x : ndarray
        Solved parameter vector
    estimator_var : ndarray
    chi2_reg : float
        The regularization penalty incurred by `x`, i.e.
        lam_eff * ||x||^2 (plain ridge) or
        lam_eff * ||reg_operator @ x||^2 (smoothness). Add this to the data-misfit chi2
        to get the total penalized objective value.
    dof_eff : float
        Effective degrees of freedom used by the fit (trace of the hat
        matrix). Use this in
        place of the nominal parameter count `len(x)` when forming a
        reduced chi2 for comparing different regularization strengths
        -- see the comment where it's computed below for why.
    dof_eff_rows : float or None
        The same quantity restricted to the rows selected by
        `dof_rows`, or None when `dof_rows` was not given.
    """
    inv_var = 1./b_var

    # Q_unreg is the *unregularized* Gram matrix of the weighted design
    # matrix: Q_unreg = A^T diag(1/b_var) A. We keep this one around
    # (never mutated) separately from the regularized `Q` used to solve
    # for `x`, because it's also exactly what's needed below to compute
    # the fit's effective degrees of freedom.
    Q_unreg = (A.T * inv_var) @ A
    print(Q_unreg.shape)

    lam_eff = 0
    Q = Q_unreg
    if lam > 0:
        # compute element wise root sum of squares
        nelements = A.shape[0]*A.shape[1]
        norm = np.sqrt(np.sum(A**2)/nelements)
        lam_eff = lam * norm
        print(f"Ridge regression {lam=}, {norm=} {nelements=} {lam_eff=}")

        if not np.isfinite(lam_eff):
            # Catch this here rather than letting it turn Q into a
            # matrix of infs/nans and failing several hundred MB of
            # allocation later inside linalg.inv. Usual cause is a
            # lambda grid whose top end overflows float64, e.g.
            # np.logspace(..., 1000, ...) -- 10**309 is already inf.
            raise ValueError(
                f"non-finite regularization strength ({lam=}, {norm=} "
                f"-> {lam_eff=}); check the lambda grid"
            )

        if reg_operator is not None:
            # Smoothness penalty: P = D^T D, so x^T P x = ||D x||^2 is
            # the sum of squared second differences (curvature) across
            # each galaxy's spectrum -- penalizes wiggles, not the
            # overall flux level.
            P = reg_operator.T @ reg_operator
            Q = Q_unreg + P * lam_eff
            # Small ridge-to-zero floor: keeps Q invertible where the
            # smoothness penalty alone leaves a flat/linear null space
            # unconstrained by data.
            diag_add = lam_eff * reg_floor
        else:
            # Plain ridge: penalize ||x||^2 directly (shrink toward zero).
            Q = Q_unreg
            diag_add = lam_eff

        # Densify (`linalg.inv` below requires it -- P*lam_eff on its
        # own can stay sparse) and add the ridge term straight onto the
        # diagonal. Done this way rather than as `Q + np.eye(n)*diag_add`
        # because that allocates a second full n x n array just to hold
        # mostly zeros -- at n=5654 that's ~256 MB of pure waste. The
        # copy is also what keeps Q_unreg unmutated, since `Q` may still
        # be an alias of it at this point.
        Q = Q.toarray() if sparse.issparse(Q) else np.array(Q, copy=True)
        Q.flat[::Q.shape[0] + 1] += diag_add

    try:
        print(f"inverting {Q.shape}")
        Qinv = linalg.inv(Q)
    except:
        raise

    rhs = (A.T * inv_var) @ b
    x = Qinv @ rhs

    if compute_var:
        # compute variance
        estimator_var = np.sum((Qinv @ Q_unreg) * Qinv, axis=1) # diag(Z @ Qinv), using Qinv symmetric
    else:
        estimator_var = None

    # Regularization contribution to chi^2: how much penalty this
    # solution incurred, on the same footing as the data-misfit chi2,
    # so the caller can report chi2_total = chi2 + chi2_reg.
    if reg_operator is not None:
        chi2_reg = lam_eff * np.sum((reg_operator @ x)**2)
    else:
        chi2_reg = lam_eff * np.sum(x**2)

    # Effective degrees of freedom used by the fit: yfit = A @ x = H @ b
    # for the "hat matrix" H = A @ Qinv @ A^T @ Cinv, and dof_eff =
    # trace(H). By the cyclic property of trace, trace(H) =
    # trace(Qinv @ A^T @ Cinv @ A) = trace(Qinv @ Q_unreg) -- cheap to
    # get since Qinv and Q_unreg are both already sitting here as small
    # (n_params x n_params) matrices, no need to touch the (much
    # larger) A again. Unlike the nominal parameter count `len(x)`,
    # dof_eff shrinks toward 0 as regularization gets stronger (fewer
    # free parameters actually used to fit the data) and grows toward
    # `len(x)` as regularization gets weaker (the fit is free to use
    # every parameter, i.e. to fit noise) -- this is what lets a
    # reduced chi2 built from dof_eff distinguish "genuinely well fit"
    # from "overfit", instead of always preferring the least
    # regularized solution.
    dof_eff = trace_product(Qinv, Q_unreg)

    # Same quantity, but summed only over the rows in `dof_rows`: the
    # hat matrix diagonal restricted to that subset of pixels. Note the
    # fit itself is unchanged -- `x` above was solved against every row.
    # This measures how much of the fit's freedom is being spent on the
    # selected pixels, so that a chi2 computed on those same pixels can
    # be turned into a GCV score with a matching complexity penalty.
    # Cheap: one Gram matrix over a row subset (sparse, and typically a
    # small fraction of the rows) plus another trace identity.
    dof_eff_rows = None
    if dof_rows is not None:
        rows = np.asarray(dof_rows)
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        A_rows = A[rows]
        Q_rows = (A_rows.T * inv_var[rows]) @ A_rows
        dof_eff_rows = trace_product(Qinv, Q_rows)

    return x, estimator_var, chi2_reg, dof_eff, dof_eff_rows


def evaluate_model(samples, frame_list, mask_list, wave_step):
    """ """
    ra, dec, wavelength = samples
    ref_flux = 1e-16 * wave_step * np.ones(len(wavelength))
    images = []
    for i, frame in enumerate(frame_list):
        counts_conversion = wave_step * frame.specframe.lineflux_to_counts(ref_flux, wavelength)
        weight = counts_conversion / (ref_flux * len(ra) * 1e16)
        # norm = counts_conversion / (ref_flux * len(ra) * 1e16)

        valid = weight > 0

        im = frame.make_image_from_samples(ra[valid], dec[valid], wavelength[valid], weight[valid],
                                           masks=mask_list[i],
                                           noise=False, return_var=False)
        # normalize to the counts that give 1e-16
        # for key in im.keys():
            # im[key] *= norm
        images.append(im)
    return images


def get_data_vec(images, var_images, mask_list, frame_ordering):
    """ """
    data_vec = []
    var_vec = []
    for (frame_i, det) in frame_ordering:
        sel = mask_list[frame_i][det][1].flatten() > -1
        data_vec.append(images[frame_i][det].flatten()[sel])
        v = var_images[frame_i][det].flatten()[sel]
        var_vec.append(v)
    data_vec = np.concatenate(data_vec)
    var_vec = np.concatenate(var_vec)
    return data_vec, var_vec


def super_concat(models, frame_ordering):
    """ """
    data = []
    for frame_i, det in frame_ordering:
        data.append(models[frame_i][det])
    return np.concatenate(data)


def compute_jacobian_11(galaxy, frame_list, mask_list, dx, i, frame_ordering):
    """ """
    wave_step = galaxy.wavelength[1] - galaxy.wavelength[0]
    ra, dec = galaxy.sample_image(int(dx*wave_step/100))
    wavelength = np.random.uniform(0, 1, len(ra)) * wave_step + galaxy.wavelength[i]
    samples = (ra, dec, wavelength)
    model = evaluate_model(samples, frame_list, mask_list, wave_step=wave_step)
    concat_model = super_concat(model, frame_ordering)
    return concat_model

def compute_jacobian_1(galaxy, frame_list, mask_list, npix, dx, frame_ordering):
    """ """
    jac = sparse.lil_array(
        (npix, len(galaxy.wavelength)),
        dtype='d'
    )

    rows, cols, vals = [], [], []
    for i in range(len(galaxy.wavelength)):
        jac_col = compute_jacobian_11(galaxy, frame_list, mask_list, dx, i, frame_ordering)
        # compute indices of sparse matrix
        nonzero_ind = np.nonzero(jac_col)[0]
        rows.append(nonzero_ind)
        cols.append(np.full(nonzero_ind.shape, i))
        vals.append(jac_col[nonzero_ind])

    vals = np.concatenate(vals)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    jac = sparse.coo_array(
        (vals, (rows, cols)),
        shape=(npix, len(galaxy.wavelength)),
    )

    return jac.tocsr()

def compute_jacobian(galaxy_list, frame_list, mask_list, frame_ordering, dx=1000, n_jobs=4):
    """Build the full sparse Jacobian mapping each galaxy's spectrum to all masked pixels.

    Computes the total number of unmasked pixels across all frames and
    detectors from `mask_list`, then builds a per-galaxy Jacobian block
    in parallel (via `compute_jacobian_1`) whose columns are the
    galaxy's wavelength samples and whose rows are the concatenated
    masked pixels (ordered per `frame_ordering`). The per-galaxy blocks
    are horizontally stacked into a single sparse matrix so that a flat
    coefficient vector over all galaxies' wavelengths maps directly onto
    the flat pixel vector produced by `get_data_vec`.

    Parameters
    ----------
    galaxy_list : list or single galaxy
        Galaxy sources to build Jacobian columns for; normalized to a
        list via `utils.ensurelist`. Each contributes
        `len(galaxy.wavelength)` columns.
    frame_list : list
        Frame/detector metadata used to evaluate the model image for
        each wavelength sample (passed through to `compute_jacobian_1`
        / `compute_jacobian_11` / `evaluate_model`).
    mask_list : list
        Per-frame dict of {detector: (npix, mask)}; determines the
        total row count and which pixels are included.
    frame_ordering : list
        Ordered (frame_i, det) pairs defining how per-detector pixel
        blocks are concatenated into rows.
    dx : int, optional
        Sampling density controlling how densely each galaxy's image is
        sampled per wavelength bin when evaluating the model (default
        1000).
    n_jobs : int, optional
        Number of parallel jobs used to compute each galaxy's Jacobian
        block (default 4).

    Returns
    -------
    jac : sparse array
        Horizontally stacked sparse Jacobian of shape
        (total masked pixels, sum of galaxies' wavelength lengths).
    """
    print(f"Compute jacobian, galaxies:{len(galaxy_list)}, frames:{len(frame_list)}, {dx=}")
    galaxy_list = utils.ensurelist(galaxy_list)

    npix = 0
    for mask in mask_list:
        for npix_, _ in mask.values():
            npix += npix_

    # jac_list = [compute_jacobian_1(gal, frame_list, mask_list, npix, dx, n_jobs_inner) for gal in tqdm(galaxy_list)]

    with parallel_config(n_jobs=n_jobs, max_nbytes=None, verbose=101):
        jac_list = Parallel()(
            delayed(compute_jacobian_1)(gal, frame_list, mask_list, npix, dx, frame_ordering) for gal in galaxy_list)

    jac = sparse.hstack(jac_list)

    return jac


def make_mask_1(specimager, galaxies, input_mask):
    """ """
    return specimager.make_mask(galaxies, input_mask=input_mask)


def unpack_extractions(pack, galaxy_list):
    """ """
    extractions = []
    i = 0
    for gal in galaxy_list:
        j = i + len(gal.wavelength)
        extractions.append(pack[i:j])
        i = j
    return extractions


def update_masklist(valid, mask_list, frame_ordering):
    """Rebuild per-detector pixel masks to account for newly invalidated pixels.

    Walks `frame_ordering` to slice the flat `valid` boolean array into
    per-(frame, detector) chunks matching each mask's pixel count. For
    each detector, pixels that were already masked (mask <= -1) or are
    now invalid are excluded, and the surviving pixels are renumbered
    with consecutive indices (``0..npix_-1``) so downstream code can index
    into the trimmed data/jac/var arrays. Each entry in `mask_list` is
    replaced with the updated ``(npix_, mask_)`` pair.

    Parameters
    ----------
    valid : ndarray of bool
        Flat validity flags, in the same concatenated order as
        `frame_ordering`, marking which pixels survive.
    mask_list : list
        Per-frame dict/list of {det: (npix, mask)}, where `mask` maps
        original pixel positions to indices (-1 for already masked).
    frame_ordering : list
        Ordered (frame_i, det) pairs describing how the flat `valid`
        array maps onto `mask_list` entries.

    Returns
    -------
    mask_list : list
        The same structure with each (npix, mask) entry replaced by
        ``(npix_, mask_)`` reflecting the newly masked/renumbered pixels.
    """
    print("updating mask list")
    mask_list_out = [{} for i in range(len(mask_list))]
    i = 0
    for frame_i, det in frame_ordering:
        npix, mask = mask_list[frame_i][det]
        j = i + npix
        valid_ = valid[i:j]
        npix_ = np.sum(valid_)
        sel, = np.where(mask.flat > -1)
        sel = sel[valid_]
        mask_ = np.zeros_like(mask) - 1
        mask_.flat[sel] = np.arange(npix_)
        mask_list_out[frame_i][det] = (npix_, mask_)
        i = j
    return mask_list_out


def mask_jacobian(jac, data, var, resid, mask_list, frame_ordering, threshold=10):
    """Flag and remove outlier pixels based on residual/sigma, and rebuild the mask list to match.

    Pixels with zero variance are treated as already masked. Among the
    remaining pixels, any whose residual exceeds `threshold` times the
    local sigma (sqrt(var)) are marked invalid. `mask_list` is then
    updated via `update_masklist` so its per-frame/detector indexing
    stays consistent with the newly masked pixels, and `jac`, `data`,
    and `var` are trimmed to the surviving (valid) rows.

    Parameters
    ----------
    jac : ndarray
        Jacobian matrix, rows aligned with `data`/`var`/`resid`.
    data : ndarray
        Data vector.
    var : ndarray
        Per-pixel variance; entries <= 0 are treated as already masked.
    resid : ndarray
        Residuals (data - model) used to compute the outlier statistic.
    mask_list : list
        Per-frame mask info, as consumed/produced by `update_masklist`.
    frame_ordering : list
        Ordering of (frame_i, det) pairs describing how `resid` maps
        onto `mask_list` entries.
    threshold : float, optional
        Number of sigma above which a pixel is masked as an outlier
        (default 10).

    Returns
    -------
    jac, data, var : ndarray
        Inputs trimmed to the valid (non-masked) rows.
    mask_list : list
        Updated mask list reflecting the newly masked pixels.
    """
    nonzero = var > 0
    valid = np.zeros(len(resid), dtype=bool)
    var_ = np.sqrt(var[nonzero])
    r = np.abs(resid[nonzero]) / var_
    print(f"residual percentiles {np.percentile(r, (50,90,95,99))}")
    valid[nonzero] = r < threshold
    nmask = np.sum(~valid)
    n = len(valid)
    print(f"masked elements {threshold=} {nmask}, {nmask*100/n}%")

    mask_list = update_masklist(valid, mask_list, frame_ordering)
    jac = jac[valid]
    data = data[valid]
    var = var[valid]
    return jac, data, var, mask_list


def target_row_mask(frame_ordering, mask_list, mask_list_target, var=None):
    """Boolean mask over the flat data vector selecting target-footprint pixels.

    Walks `frame_ordering` in the same order `get_data_vec` used to
    concatenate the data, and for each (frame, detector) block marks the
    surviving pixels that also fall inside the target sources' footprint
    according to `mask_list_target`. Detectors absent from
    `mask_list_target` (no target lands on them) contribute an all-False
    block. If `var` is given, pixels with non-positive variance are
    dropped as well, matching what `compute_chi2_target` counts.

    Factored out so that the target chi2 and the target-restricted
    effective dof are guaranteed to refer to the identical set of
    pixels -- a GCV score built from a chi2 and a trace computed over
    different pixel sets would be silently wrong.

    Parameters
    ----------
    frame_ordering : list
        Ordered (frame_i, det) pairs, as built by `measure_everything`.
    mask_list : list
        Current per-frame {det: (npix, mask)} masks, i.e. the ones that
        match the *current* (post-sigma-clipping) data vector.
    mask_list_target : list
        Per-frame {det: (npix, mask)} masks built from the target
        galaxies alone.
    var : ndarray, optional
        Variance vector aligned with the data vector; when given,
        entries <= 0 are excluded.

    Returns
    -------
    sel : ndarray of bool
        Length equals the current data vector length (sum of `npix`
        over `frame_ordering`).
    """
    blocks = []
    for frame_i, det in frame_ordering:
        npix_, mask = mask_list[frame_i][det]
        try:
            _, target_mask = mask_list_target[frame_i][det]
        except KeyError:
            blocks.append(np.zeros(npix_, dtype=bool))
            continue
        # `mask > -1` are exactly the pixels this block contributed to
        # the data vector, in order, so indexing the target mask by it
        # lines the two up row for row.
        blocks.append(target_mask[mask > -1] > -1)

    sel = np.concatenate(blocks) if blocks else np.zeros(0, dtype=bool)

    if var is not None:
        sel &= var > 0

    return sel


def compute_chi2_target(frame_ordering, mask_list, mask_list_target, data, model, var, sel=None):
    """Chi2, rms and pixel count over the target sources' footprint only.

    Pass `sel` (from `target_row_mask`) to reuse a mask already computed
    by the caller; otherwise it is derived here.
    """
    if sel is None:
        sel = target_row_mask(frame_ordering, mask_list, mask_list_target, var)

    resid = data[sel] - model[sel]
    npix = int(np.sum(sel))
    chi2 = np.sum(resid**2 / var[sel])
    rms = np.sqrt(np.sum(resid**2)/npix) if npix > 0 else 0
    return chi2, rms, npix


def measure_everything(images, var_images, frame_list, target_galaxies,contaminant_galaxies=[], pixmask_list=None, jac=None, mask_list=None, dx=10000, n_jobs=4, masking_iterations=1, sigma_clip=5, smooth=True, reg_floor=1e-3, reg_lambda_list=[1], compute_var=False):
    """Extract spectra for all galaxies by building/solving the joint linear model across frames.

    Builds per-detector masks (via `make_mask_1`) if `mask_list` is not
    given, reconciles `images` and `mask_list` so both cover exactly the
    same frame/detector entries, and derives `frame_ordering` describing
    how per-detector pixels are concatenated. Builds the sparse Jacobian
    (via `compute_jacobian`) if not supplied, and assembles the flat
    data/variance vectors (via `get_data_vec`).

    For each regularization strength in `reg_lambda_list`, iteratively
    solves the linear system (via `solve`), computing residuals and
    optionally re-masking outlier pixels (via `mask_jacobian`) for up to
    `masking_iterations` rounds before finalizing chi2/dof and unpacking
    the fitted coefficients into per-galaxy SED extractions.

    Parameters
    ----------
    images : list
        Per-frame dict of {detector: image array} to extract from.
    var_images : list
        Per-frame dict of {detector: variance array}, aligned with
        `images`.
    frame_list : list
        Frame/detector metadata used to build masks and the Jacobian.
    galaxies : list
        Galaxy sources to extract; each carries its own `wavelength`
        grid.
    pixmask_list : list, optional
        Per-frame input pixel masks passed to `make_mask_1` when
        `mask_list` is not supplied.
    jac : sparse array, optional
        Precomputed Jacobian; if None, it is built via
        `compute_jacobian`.
    reg_lambda_list : list, optional
        Regularization strengths to solve for; one result is returned
        per entry (default [1]).
    smooth : bool, optional
        If True, regularize with a curvature (second-difference)
        penalty via `build_smoothness_operator`/`solve`'s
        `reg_operator`, instead of the plain ridge (shrink-to-zero)
        penalty (default False).
    mask_list : list, optional
        Precomputed per-frame/detector masks; if None, built via
        `make_mask_1` from `pixmask_list`.
    dx : int, optional
        Sampling density passed to `compute_jacobian` (default 10000).
    n_jobs : int, optional
        Number of parallel jobs used for mask building and Jacobian
        computation (default 4).
    masking_iterations : int, optional
        Number of solve/re-mask rounds performed via `mask_jacobian`
        before the final solve for each lambda (default 1).

    Returns
    -------
    results : list of (galaxies_out, fit_info)
        For each lambda in `reg_lambda_list`: `galaxies_out` is a copy of
        `galaxies` with `.sed` set to the extracted spectrum, and
        `fit_info` is a dict with `frame_ordering`, `data`, `model`,
        `resid`, `var`, `chi2`, `dof`, `jac`, `mask_list`, and `lam`.

        No lambda is selected here -- every entry in `reg_lambda_list`
        is returned and the choice is left to the caller. `fit_info`
        carries two generalized cross-validation scores for that
        purpose, each to be minimized over the returned results:

        `gcv_targ` -- computed over the target sources' footprint only,
        and the one to select on. It scores the fit where it matters,
        on the pixels the extracted target spectrum is actually read
        from.

        `gcv` -- the same score over every fitted pixel. Kept as a
        diagnostic; it is dominated by contaminant and background
        pixels, so its minimum can sit at a different lambda.

        See where they are computed below for the assumptions behind
        both.
    """
    print(f"{masking_iterations=}")
    galaxies = target_galaxies + contaminant_galaxies
    with parallel_config(n_jobs=n_jobs, max_nbytes=None, verbose=101):
        if mask_list is None:
            print(f"Building masks")
            mask_list = Parallel()(delayed(make_mask_1)(
                        frame_list[frame_i], galaxies, pixmask_list[frame_i]) for frame_i in range(len(frame_list)))
        mask_list_target = Parallel()(delayed(make_mask_1)(
            frame_list[frame_i], target_galaxies, pixmask_list[frame_i]) for frame_i in range(len(frame_list)))

    # ensure that data and mask lists contain the same detectors
    print("checking images and masks")
    for i in range(len(images)):
        for detector in list(images[i].keys()):
            if detector not in mask_list[i]:
                print(f"missing mask frame {i} det {detector}")
                del images[i][detector]

    for i in range(len(mask_list)):
        for detector in list(mask_list[i].keys()):
            if detector not in images[i]:
                print(f"missing image frame {i} det {detector}")
                del mask_list[i][detector]
    print("ok")

    frame_ordering = []
    for frame_i in range(len(mask_list)):
        for det in images[frame_i].keys():
            frame_ordering.append((frame_i, det))

    data, var = get_data_vec(images, var_images, mask_list, frame_ordering)

    if jac is None:
        print('Building jacobian...')
        t0 = time.time()
        jac = compute_jacobian(galaxies, frame_list, mask_list, frame_ordering, dx=dx, n_jobs=n_jobs)
        print(f"jacobian done: {time.time()-t0} sec")

    print('Start solving...')

    # Build the smoothness regularization operator once, up front: it only
    # depends on each galaxy's wavelength grid (via `galaxy_list`), not on
    # the data or on `lam`, so there's no need to rebuild it per lambda or
    # per masking iteration. `smooth` now selects curvature-penalty
    # regularization (see `build_smoothness_operator`) instead of the old
    # post-hoc Gaussian-kernel smoothing.
    reg_operator = build_smoothness_operator(galaxies) if smooth else None

    results = []
    for lam in reg_lambda_list:
        data_ = data.copy()
        var_ = var.copy()
        jac_ = jac.copy()
        mask_list_ = update_masklist(np.ones(len(data), dtype=bool), mask_list, frame_ordering)

        for iteration in range(masking_iterations + 1):
            # Which rows of the current data vector lie in the target
            # footprint. Recomputed every iteration because sigma-clipping
            # trims rows out from underneath it; on the final iteration
            # `mask_list_`/`var_` are no longer touched, so this is the
            # same selection `compute_chi2_target` scores below.
            target_rows = target_row_mask(frame_ordering, mask_list_, mask_list_target, var_)
            compute_var_ = compute_var and (iteration == masking_iterations)
            x, estimator_var, chi2_reg, dof_eff, dof_eff_targ = solve(
                jac_, data_, var_, lam=lam, reg_operator=reg_operator,
                reg_floor=reg_floor,
                dof_rows=target_rows,
                compute_var=compute_var_
            )
            yfit = jac_ @ x
            resid = yfit - data_
            if iteration < masking_iterations:
                print(f"masking jacobian iteration {iteration+1}")
                jac_, data_, var_, mask_list_ = mask_jacobian(
                    jac_, data_, var_, resid, mask_list_, frame_ordering, threshold=sigma_clip)

        chi2_targ, rms_targ, npix_targ = compute_chi2_target(
            frame_ordering, mask_list_, mask_list_target,
            data_, yfit, var_, sel=target_rows
        )

        chi2 = np.sum(resid**2 / var_)
        chi2_total = chi2 + chi2_reg
        rms = np.sqrt(np.mean(resid**2))
        dof = len(x)
        npix = len(data_)
        # Reduced chi2 using the *effective* degrees of freedom rather than
        # the raw pixel count npix. Raw chi2/npix always improves (goes
        # down) as regularization weakens, since a less-constrained fit can
        # always get closer to the data -- including its noise. dof_eff
        # tracks how many parameters the fit is effectively free to use,
        # so npix - dof_eff shrinks as regularization weakens too, and this
        # ratio can properly go back up for an overfit (too weakly
        # regularized) solution instead of monotonically favoring it.
        residual_dof_eff = npix - dof_eff
        reduced_chi2_eff = chi2 / residual_dof_eff if residual_dof_eff > 0 else np.inf
        # Generalized cross-validation score: the standard closed-form
        # approximation to leave-one-out CV for a linear smoother, which
        # this fit is (yfit = H @ data for the hat matrix H whose trace is
        # dof_eff). Rather than refitting with pixels held out, it rescales
        # the in-sample chi2 by (1 - tr(H)/npix)^-2:
        #
        #     gcv = (chi2/npix) / (1 - dof_eff/npix)^2
        #         = npix * chi2 / residual_dof_eff^2
        #
        # Pick the lambda that minimizes it. Like reduced_chi2_eff it
        # penalizes weak regularization via dof_eff, but the squared
        # denominator makes it the sharper of the two -- reduced_chi2_eff
        # can stay nearly flat across a wide lambda range where gcv shows a
        # clear minimum. Caveat: GCV assumes independent residuals, and
        # here neighbouring pixels are correlated by the PSF and the
        # dispersion, so tr(H) understates the true model complexity and
        # the minimum tends to sit at somewhat weaker regularization than
        # a true held-out CV would choose. Treat it as a starting point,
        # not a final answer.
        gcv = npix * chi2 / residual_dof_eff**2 if residual_dof_eff > 0 else np.inf
        # The same GCV score restricted to the target sources' own pixels
        # -- this is the one to select lambda on. The global `gcv` above
        # is dominated by contaminant and background pixels, which vastly
        # outnumber the target's; the lambda that best predicts those is
        # not necessarily the lambda that best recovers the target
        # spectrum, which is the only output anyone consumes. Both the
        # chi2 and the complexity penalty are restricted to the same
        # pixels (`target_rows`), so the score stays a valid GCV: dof_eff
        # counts the freedom the fit spends *on the target footprint*,
        # which is far smaller than the total, and shrinks with lambda in
        # the same way. The independent-residual caveat above still
        # applies, and applies more strongly here -- the target footprint
        # is a small, contiguous, PSF-correlated patch.
        residual_dof_eff_targ = npix_targ - dof_eff_targ
        gcv_targ = (npix_targ * chi2_targ / residual_dof_eff_targ**2
                    if residual_dof_eff_targ > 0 else np.inf)
        print(f"{chi2_reg=}")
        print(f"{lam=} {chi2_targ=} {rms_targ=} {npix_targ=}")
        print(f"{lam=} {chi2=} {rms=} {npix=} {dof=}")
        print(f"{lam=} {dof_eff=} {residual_dof_eff=} {reduced_chi2_eff=} {gcv=}")
        print(f"{lam=} {dof_eff_targ=} {residual_dof_eff_targ=} {gcv_targ=}")

        extractions = unpack_extractions(x, galaxies)

        galaxies_out = []
        for i in range(len(galaxies)):
            new_gal = galaxies[i].copy()
            new_gal.sed = extractions[i] * 1e-16
            galaxies_out.append(new_gal)

        fit_info = dict(estimator_var=estimator_var, frame_ordering=frame_ordering, data=data_, model=yfit, resid=resid, var=var_,
                        chi2_total=chi2_total, chi2_reg=chi2_reg,
                        chi2=chi2,  rms=rms, npix=npix, dof=dof,
                        dof_eff=dof_eff, reduced_chi2_eff=reduced_chi2_eff, gcv=gcv,
                        chi2_targ=chi2_targ, rms_targ=rms_targ, npix_targ=npix_targ,
                        dof_eff_targ=dof_eff_targ, gcv_targ=gcv_targ,
                        jac=jac_, mask_list=mask_list_, mask_list_target=mask_list_target, lam=lam, sigma_clip=sigma_clip)
        results.append((galaxies_out, fit_info))

    return results
