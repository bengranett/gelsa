import numpy as np


def check_overlap(mask1, mask2):
    """ """
    for d in mask1.keys():
        sel1 = mask1[d] > -1
        if d in mask2:
            sel2 = mask2[d] > -1
            if np.any(sel1 & sel2):
                return True
    return False



def get_overlap_galaxies_(gal_list, target_gal, frame_list):
    """ """
    cache = {}
    sel = [target_gal]
    cache[target_gal.params['id']] = 1

    for S in frame_list:
        mask_targ = S.make_mask([target_gal], width=1)

        for gal in gal_list:
            key = gal.params['id']
            if key in cache:
                continue

            m = S.make_mask([gal])

            if check_overlap(mask_targ, m):
                sel.append(gal)
                print(sel)

    print(f"Number of overapping sources: {len(sel)}")
    return sel


def get_overlap_galaxies(gal_list, target_gal, frame_list):
    """ """
    cache = {}
    sel = [target_gal]
    cache[target_gal.params['id']] = 1

    for S in frame_list:
        for gal in gal_list:
            key = gal.params['id']
            if key in cache:
                continue
            if S.check_overlap(target_gal, gal):
                sel.append(gal)
                cache[key] = 1
    print(f"Number of overapping sources: {len(sel)}")
    return sel


def check_segment_overlap(x1, y1, x2, y2, separation=10):
    """Check if two segements overlap. Checks every pair of points
    if closer than separation.

    Parameters
    ----------
    x1
    y1
    x2
    y2
    separation

    Returns
    -------
    bool
    """
    x1 = np.array(x1)
    y1 = np.array(y1)
    x2 = np.array(x2)
    y2 = np.array(y2)
    dx2 = x1[:,np.newaxis]**2 + x2**2 - 2*np.outer(x1, x2)
    dy2 = y1[:,np.newaxis]**2 + y2**2 - 2*np.outer(y1, y2)
    sep2 = dx2 + dy2
    # print(f"min sep {np.sqrt(np.min(sep2))}")
    return np.any(sep2 < separation**2)
