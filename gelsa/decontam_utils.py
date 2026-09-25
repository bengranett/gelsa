import numpy as np
import healpy
from matplotlib import pyplot as plt

from gelsa import galaxy, spec_imager


def pack_images(frame_list, gal_list):
    """ """
    specimager_list = []

    images = []
    var_images = []
    pix_mask = []

    pointing_id_list = {}

    for frame in frame_list:
        hit = False
        pointing_id = frame.params['PTGID']
        if pointing_id in pointing_id_list:
            print(f"duplicate pointing ID! {pointing_id}")
            frame.close_frame()
            continue
        pointing_id_list[pointing_id] = 1
        wave_range = frame.params['wavelength_range']
        S = spec_imager.SpectralImager(frame)
        image_ = {}
        var_ = {}
        mask_ = {}
        for gal in gal_list:
            for wave in np.linspace(*wave_range, 100):
                x, y, det = frame.radec_to_pixel(
                    gal.params['ra'], gal.params['dec'], wave)
                if det in image_:
                    continue
                if det < 0:
                    continue
                im, mask, var = frame.get_detector(det)
                valid = (mask == 0) & np.isfinite(im) & np.isfinite(var) & (var > 0)
                ### watch out
                valid &= (im > -1000) & (im < 10_000)
                ####
                input_mask = np.ones(mask.shape, dtype=bool)
                input_mask[~valid] = 0
                # # replace variance with constant
                # medvar = np.median(var[valid])
                # print(f"median var {det} {medvar}")
                # var[:,:] = medvar
                im[~valid] = 0
                var[~valid] = 0
                image_[det] = im
                var_[det] = var
                mask_[det] = input_mask
                hit = True
        frame.close_frame()
        if hit:
            specimager_list.append(S)
            images.append(image_)
            var_images.append(var_)
            pix_mask.append(mask_)
    return specimager_list, images, var_images, pix_mask


def get_overlapping_sources(frame,
                            ra_center, dec_center,
                            cat_ra, cat_dec,
                            separation_arcsec=5,
                            dispersion_ang_pix=13):
    """ """
    wave_low, wave_high = frame.params['wavelength_range']
    width_lambda = (wave_high - wave_low)
    width_arcsec = width_lambda / dispersion_ang_pix * frame.params['PIXSCALE']

    theta = -frame.params['PA'] + frame.params['tilt']
    R = healpy.Rotator((ra_center, dec_center, theta))
    dx, dy = R(cat_ra, cat_dec, lonlat=True)
    separation_dispersion = width_arcsec / 3600
    separation = separation_arcsec / 3600
    distance = np.sqrt(dx**2 + dy**2)
    sel = (np.abs(dx) < separation_dispersion) & \
          (np.abs(dy) < separation)
    return sel, distance


def build_contaminant_list(frame_list, target_list, cat, separation_arcsec=5, minimum_separation_arcsec=0.3):
    """ """
    cat_ra = cat['RIGHT_ASCENSION']
    cat_dec = cat['DECLINATION']

    select = np.zeros(len(cat), dtype=bool)
    exclude = np.zeros(len(cat), dtype=bool)
    for frame in frame_list:
        for target in target_list:
            sel_, distance = get_overlapping_sources(
                frame,
                target['ra'], target['dec'],
                cat_ra, cat_dec,
                separation_arcsec=separation_arcsec,
            )
            select |= sel_
            exclude |= distance < minimum_separation_arcsec / 3600

    # also exclude by object ID
    target_object_ids = [t['object_id'] for t in target_list if t['object_id'] is not None]
    exclude |= np.isin(cat['OBJECT_ID'], target_object_ids)

    return cat[select & ~exclude]


def find_image(image_list, ra, dec, tile=None):
    """ """
    for image_pack in image_list:
        # use tile index to locate the image
        this_tile = image_pack.get('tile', None)
        if tile and this_tile:
            if this_tile == tile:
                return image_pack
            else:
                continue
        # if tile is not known, use wcs
        wcs = image_pack['wcs']
        shape = wcs.array_shape
        x, y = wcs.all_world2pix(ra, dec, 0)
        if (x < 0) | (y < 0):
            continue
        if (x >= shape[1]) | (y >= shape[0]):
            continue
        return image_pack
    return None


def crop_image(ra, dec, segmap_id, image=None, segmap=None, wcs=None, tile=None, width=101):
    """ """
    x, y = wcs.all_world2pix(ra, dec, 0)
    if segmap_id is None:
        segmap_id = segmap[int(y), int(x)]
    w = width // 2
    ny, nx = image.shape
    xmin = max(0, int(x-w))
    xmax = min(nx, int(x+w))
    ymin = max(0, int(y-w))
    ymax = min(ny, int(y+w))
    sel = np.s_[ymin:ymax, xmin:xmax]
    crop = image[sel].copy()
    crop_segmap = segmap[sel].copy()
    crop_wcs = wcs[sel]
    target = crop_segmap == segmap_id
    # if np.sum(target) == 0:
        # print("soemthing went wrong with the segmap", segmap_id)
    crop[~target] = 0
    crop_segmap[~target] = 0
    return dict(image=crop, segmap=crop_segmap, wcs=crop_wcs, tile=tile)


def build_gal_list(cat, image_list,
                   wavelength_range=(9200,19000),
                   wavelength_step=100):
    """ """
    gal_list = []

    for row in cat:
        image_pack = find_image(image_list,
                                row['RIGHT_ASCENSION'],
                                row['DECLINATION'],
                                row['TILE_INDEX']
        )
        if image_pack is None:
            continue

        crop_pack = crop_image(
            row['RIGHT_ASCENSION'], row['DECLINATION'],
            row['SEGMENTATION_MAP_ID'],
            **image_pack
        )
        if np.sum(crop_pack['image']) == 0:
            # print("crop is empty")
            continue

        gal = galaxy.Galaxy(
            object_id=row['SEGMENTATION_MAP_ID'],
            ra=row['RIGHT_ASCENSION'],
            dec=row['DECLINATION'],
            redshift=0,
            profile='image',
            obs_wavelength_step=wavelength_step,
            obs_wavelength_range=wavelength_range,
        )
        gal.set_image(crop_pack['image'], crop_pack['segmap'], crop_pack['wcs'])
        gal_list.append(gal)

    return gal_list

def build_target_gal_list(target_list, image_list,
                          wavelength_range=(9200, 19000),
                          wavelength_step=100):
    """ """
    gal_list = []
    for target_pack in target_list:
        crop_pack = None
        if ('image' in target_pack) and (target_pack['image'] is not None):
            crop_pack = dict(
                image=target_pack['image'],
                segmap=target_pack['segmap'],
                wcs=target_pack['wcs']
            )
        else:
            image_pack = find_image(image_list,
                                    target_pack['ra'],
                                    target_pack['dec'],
                                    target_pack['tile']
            )
            if image_pack is None:
                raise ValueError(f"Could not find image for target {target_pack}")
            crop_pack = crop_image(
                target_pack['ra'], target_pack['dec'],
                target_pack['segmap_id'], **image_pack
            )
            if np.sum(crop_pack['image']) == 0:
                raise ValueError(f"no crop for target {target_pack}")

        if target_pack['segmap_id'] is None:
            x, y = crop_pack['wcs'].all_world2pix(target_pack['ra'], target_pack['dec'], 0)
            target_pack['segmap_id'] = crop_pack['segmap'][int(y), int(x)]
                
        gal = galaxy.Galaxy(
            object_id=target_pack['segmap_id'],
            ra=target_pack['ra'],
            dec=target_pack['dec'],
            redshift=0,
            profile='image',
            obs_wavelength_step=wavelength_step,
            obs_wavelength_range=wavelength_range,
        )
        gal.set_image(crop_pack['image'], crop_pack['segmap'], crop_pack['wcs'])
        gal_list.append(gal)
    return gal_list



def plot2d(fit_list, vmin=-50, vmax=200, cmap='inferno_r',
          only_target=False, label="", tag='all', outdir=None,
              pad=10):
    """ """
    frame_ordering = fit_list['frame_ordering']
    mask_list = fit_list['mask_list']
    data = fit_list['data']
    resid = fit_list['resid']
    model = fit_list['model']

    i = 0
    out_list = []
    for frame_i, det in frame_ordering:
        npix, mask = mask_list[frame_i][det]
        j = i + npix
        d = data[i:j]
        m = model[i:j]
        r = resid[i:j]
        i = j

        shape = mask.shape
        sel = mask > -1
        im_d = np.zeros(shape, dtype='d')
        im_m = np.zeros(shape, dtype='d')
        im_r = np.zeros(shape, dtype='d')
        im_d[sel] = d
        im_m[sel] = m
        im_r[sel] = r

        rows, cols = np.nonzero(im_d)
        ymin, ymax = rows.min(), rows.max()
        xmin, xmax = cols.min(), cols.max()

        xmin = max(0, xmin-pad)
        xmax = min(shape[1], xmax+pad)
        ymin = max(0, ymin-pad)
        ymax = min(shape[0], ymax+pad)

        im_d[~sel] = np.nan
        im_m[~sel] = np.nan
        im_r[~sel] = np.nan

        im_d = im_d[ymin:ymax, xmin:xmax]
        im_r = im_r[ymin:ymax, xmin:xmax]
        im_m = im_m[ymin:ymax, xmin:xmax]

        shape = im_m.shape
        aspect = shape[0]/shape[1]
        plt.figure(figsize=(8,8*aspect*2))
        ax1=plt.subplot(311)
        plt.imshow(im_d, vmin=vmin, vmax=vmax, cmap=cmap)
        ax1.text(0,1,"Data", transform=ax1.transAxes,c='k', va='top', fontsize=14)
        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticklabels([])
        plt.colorbar(pad=0,  aspect=5)

        ax2=plt.subplot(312)
        plt.imshow(im_m, vmin=vmin, vmax=vmax, cmap=cmap)
        ax2.text(0,1,"GELSA model", transform=ax2.transAxes,c='k', va='top', fontsize=14)
        ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])
        plt.colorbar(pad=0, aspect=5)

        ax3=plt.subplot(313)
        plt.imshow(im_r, vmin=-vmax, vmax=vmax, cmap='bwr')
        ax3.text(0,1,"Residual", transform=ax3.transAxes,c='k', va='top', fontsize=14)
        ax3.xaxis.set_ticklabels([])
        ax3.yaxis.set_ticklabels([])
        plt.colorbar(pad=0, aspect=5)

        plt.subplots_adjust(left=0.02, right=0.97, bottom=.05, top=0.95, hspace=0.05)
        if outdir:
            plt.savefig(outdir + f"spec2d_{label}_{tag}_frame{frame_i}_det{det}.png")
            plt.close()


def plot2d_target(fit_list, frame_list, fig=None, vmin=-50, vmax=200, cmap='inferno_r',
          only_target=False, label="", tag='target', outdir=None,
              pad=10):
    """ """
    frame_ordering = fit_list['frame_ordering']
    mask_list = fit_list['mask_list']
    mask_list_target = fit_list['mask_list_target']
    data = fit_list['data']
    resid = fit_list['resid']
    model = fit_list['model']
    var = fit_list['var']
    chi2 = np.zeros_like(resid)
    valid = var > 0
    chi2[valid] = resid[valid]**2 / var[valid]

    i = 0
    out_list = []
    for frame_i, det in frame_ordering:
        skip = False
        params = frame_list[frame_i].specframe.params

        npix, mask = mask_list[frame_i][det]
        j = i + npix

        try:
            _, target_mask = mask_list_target[frame_i][det]
        except KeyError:
            i = j
            continue

        valid = (mask > -1)
        targ_valid = target_mask > -1
        sel = target_mask[valid] > -1
        if np.sum(sel) == 0:
            i = j
            continue

        d = data[i:j][sel]
        m = model[i:j][sel]
        r = resid[i:j][sel]
        q = chi2[i:j][sel]

        reduced_chi2 = np.mean(q)

        i = j

        shape = mask.shape

        im_d = np.zeros(shape, dtype='d') + np.nan
        im_m = np.zeros(shape, dtype='d') + np.nan
        im_r = np.zeros(shape, dtype='d') + np.nan
        im_d[valid & targ_valid] = d
        im_m[valid & targ_valid] = m
        im_r[valid & targ_valid] = r

        zeros = ~ np.isfinite(im_d)
        im_d[zeros] = 0
        im_m[zeros] = 0
        im_r[zeros] = 0

        rows, cols = np.nonzero(im_d)

        ymin, ymax = rows.min(), rows.max()
        xmin, xmax = cols.min(), cols.max()

        xmin = max(0, xmin-pad)
        xmax = min(shape[1], xmax+pad)
        ymin = max(0, ymin-pad)
        ymax = min(shape[0], ymax+pad)

        im_d[zeros] = np.nan
        im_m[zeros] = np.nan
        im_r[zeros] = np.nan

        im_d = im_d[ymin:ymax, xmin:xmax]
        im_r = im_r[ymin:ymax, xmin:xmax]
        im_m = im_m[ymin:ymax, xmin:xmax]

        shape = im_m.shape
        aspect = shape[0]/shape[1]
        if fig is None:
            plt.figure(figsize=(8,16*aspect))

        info = f"Pointing ID: {params['PTGID']}\nDither {params['DITHOBS']}\nGrism: {params['grism_name']}\nDetector: {det}\nchi$^{2}$/N={reduced_chi2:g}"


        plt.figtext(0.01,0.01, info, ha='left', va='bottom')
        ax1=plt.subplot(311)
        plt.imshow(im_d, vmin=vmin, vmax=vmax, cmap=cmap)
        ax1.text(0,1,"Data", transform=ax1.transAxes,c='k', va='top', fontsize=14)
        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticklabels([])
        plt.colorbar(pad=0,  aspect=5)

        ax2=plt.subplot(312)
        plt.imshow(im_m, vmin=vmin, vmax=vmax, cmap=cmap)
        ax2.text(0,1,"GELSA model", transform=ax2.transAxes,c='k', va='top', fontsize=14)
        ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])
        plt.colorbar(pad=0, aspect=5)

        ax3=plt.subplot(313)
        plt.imshow(im_r, vmin=-vmax, vmax=vmax, cmap='bwr')
        ax3.text(0,1,"Residual", transform=ax3.transAxes,c='k', va='top', fontsize=14)
        ax3.xaxis.set_ticklabels([])
        ax3.yaxis.set_ticklabels([])
        plt.colorbar(pad=0, aspect=5)

        plt.subplots_adjust(left=0.02, right=0.97, bottom=.05, top=0.95, hspace=0.05)
        if outdir:
            plt.savefig(outdir + f"spec2d_{label}_{tag}_frame{frame_i}_det{det}.png")
            plt.close()
