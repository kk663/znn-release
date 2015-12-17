#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""

import numpy as np
from front_end import znetio
import shutil

def timestamp():
    import datetime

    current_time = str(datetime.datetime.now())

    whitespace_removed = current_time.replace(' ','_')
    condensed_date = whitespace_removed.replace('-','')
    truncated_time = condensed_date.split('.')[0].replace(':','')

    return truncated_time

def write_to_log(filename, line):
    '''Writes a line of output to the log file. Manages opening and closing'''
    with open(filename, 'a') as f:
        f.write(line)
        f.write('\n')
        f.close()

def rft_to_string(rft):
    '''Transforms an rft (bool array) into a string for logging'''
    if rft is None:
	return "[]"

    rft_mapping = ["z-reflection", "y-reflection",
		   "x-reflection", "xy-transpose"]

    rft_matches_mapping = len(rft) == len(rft_mapping)
    assert(rft_matches_mapping)

    applied_rules = [rft_mapping[i]
	for i in range(len(rft_mapping)) if rft[i]]

    rft_string = applied_rules.__repr__().replace("'","")

    return rft_string

def data_aug_transform(data, rft):
        """
        transform data according to a rule

        Parameters
        ----------
        data : 3D numpy array need to be transformed
        rft : transform rule, specified as an array of bool
            [z-reflection,
            y-reflection,
            x-reflection,
            xy transpose]

        Returns
        -------
        data : the transformed array
        """

        if np.size(data)==0 or np.size(rft)==0:
            return data

        #z-reflection
        if rft[0]:
            data  = data[:, ::-1, :,    :]
        #y-reflection
        if rft[1]:
            data  = data[:, :,    ::-1, :]
        #x-reflection
        if rft[2]:
            data = data[:,  :,    :,    ::-1]
        # transpose in XY
        if rft[3]:
            data = data.transpose(0,1,3,2)

        return data

def _mirror2d( im, bf, fov ):
    """
    mirror image in 2D

    Parameters
    ----------
    im : 2D array
    bf : buffer for mirrored image
    fov : 2D vector

    Returns
    -------
    bf : mirrored buffer
    """
    bsz = np.asarray(bf.shape, dtype='int')
    isz = np.asarray(im.shape, dtype='int')
    fov = fov.astype('int32')
    l = (fov-1)/2
    b = bsz - (fov/2)
    i = isz - (fov/2)

    # 4 edges
    bf[:l[0], l[1]:b[1]] = im[:l[0], :][::-1, :]
    bf[l[0]:b[0], :l[1]] = im[:, :l[1]][:, ::-1]

    bf[b[0]:, l[1]:b[1]] = im[i[0]:, :][::-1, :]
    bf[l[0]:b[0], b[1]:] = im[:, i[1]:][:, ::-1]

    # 4 corners
    bf[:l[0], :l[1]] = im[:l[0], :l[1]][::-1,::-1]
    bf[b[0]:, b[1]:] = im[i[0]:, i[1]:][::-1,::-1]
    bf[:l[0], b[1]:] = im[:l[0], i[1]:][::-1,::-1]
    bf[b[0]:, :l[1]] = im[i[0]:, :l[1]][::-1,::-1]
    return bf

def boundary_mirror( arr, fov ):
    """
    mirror the boundary for each 3d array

    Parameters
    ----------
    arr : 4D array.
    fov : vector with 3 int number, field of view.

    Return
    ------
    ret : expanded 4D array with mirrored boundary
    """
    assert(np.size(fov)==3)
    print "boundary mirror..."
    fov = fov.astype('int32')
    if np.all(fov==1):
        return arr
    # buffer size
    bfsz = np.asarray(arr.shape, dtype='int32')
    bfsz[1:] += fov-1
    # initialize the buffer
    bf = np.zeros(tuple(bfsz), dtype=arr.dtype)

    # low and high of fov
    l = (fov-1)/2
    b = bfsz[1:] - fov/2
    # fill the buffer with existing array
    bf[:, l[0]:b[0], l[1]:b[1], l[2]:b[2]] = arr
    for c in xrange(arr.shape[0]):
        for z in xrange(arr.shape[1]):
            bf[c,z+l[0],:,:] = _mirror2d(arr[c, z, :, :], bf[c,z+l[0],:,:], fov[1:])
        for y in xrange(arr.shape[2]):
            bf[c,:,y+l[1],:] = _mirror2d(arr[c, :, y, :], bf[c,:,y+l[1],:], fov[0:3:2])
        for x in xrange(arr.shape[3]):
            bf[c,:,:,x+l[2]] = _mirror2d(arr[c, :, :, x], bf[c,:,:,x+l[2]], fov[:2])

        # repeat mirroring z sections for filling 8 corners
        for z in xrange(l[0]):
            bf[c,z,:,:] = _mirror2d(bf[c, z, l[1]:b[1], l[2]:b[2]], bf[c,z,:,:], fov[1:])
        for z in xrange(b[0],bfsz[1]):
            bf[c,z,:,:] = _mirror2d(bf[c, z, l[1]:b[1], l[2]:b[2]], bf[c,z,:,:], fov[1:])
    return bf

def make_continuous( d , dtype='float32'):
    """
    make the dictionary arrays continuous.

    Parameters
    ----------
    d : dict, the input dictionary of 4D array.

    Returns
    -------
    d : dict, the inner array are continuous.
    """
    for name, arr in d.iteritems():
        d[name] = np.ascontiguousarray(arr, dtype=dtype)
    return d

def get_vox_num( d ):
    n = 0
    for name, arr in d.iteritems():
        n = n + arr.shape[0]*arr.shape[1]*arr.shape[2]*arr.shape[3]
    return n

def get_total_num(outputs):
    """
    """
    n = 0
    for name, sz in outputs.iteritems():
        n = n + np.prod(sz)
    return n

def sum_over_dict(dict_vol):
    s = 0
    for name, vol in dict_vol.iteritems():
        s += vol.sum()
    return s

def dict_mask_empty(mask):
    vals = mask.values()
    return all([val.size == 0 for val in vals])

def dict_mul(das,dbs):
    """
    multiplication of two dictionary
    """
    ret = dict()
    for name, a in das.iteritems():
        b = dbs[name]
        if b.shape==a.shape:
            ret[name] = a * b
        elif np.size(b)==0:
            ret[name] = a
    return ret

def save_grdts( grdts, fname ):
    """
    save gradients
    the gradient array was stored in a dictionary
    """
    assert len(grdts.keys()) == 1
    import h5py
    f = h5py.File( fname, 'a' )
    f.create_dataset('/processing/znn/train/gradient/original_gradient', \
                     data=grdts.values()[0])
    f.close()

def save_malis( mws, fname ):
    """
    save malis weights
    the weights was stored in a dictionary
    """
    assert len(mws.keys()) == 1
    import h5py
    f = h5py.File( fname, 'a' )
    f.create_dataset('/processing/znn/train/gradient/malis_weight', \
                     data=mws.values()[0])
    f.close()

def save_rebalance( rws, fname ):
    """
    save rebalance weights
    the weights was stored in a dictionary
    """
    assert len(rws.keys())==1
    import h5py
    f = h5py.File( fname, 'a' )
    f.create_dataset('/processing/znn/train/gradient/rebalance_weight', \
                     data=rws.values()[0])
    f.close()

def get_malis_cost( props, lbl_outs, malis_weights ):
    assert( len(props.keys()) == 1 )

    # dictionary of malis weighted pixel classification error
    dmc = dict()
    # dictionary of malis weighted binomial cross entropy energy
    dme = dict()
    for key, mw in malis_weights.iteritems():
        prop = props[key]
        lbl = lbl_outs[key]
        cls = ( (prop>0.5)!=(lbl>0.5) )
        cls = cls.astype('float32')
        # cost energy
        eng = -lbl*np.log(prop) - (1-lbl)*np.log(1-prop)

        dmc[key] = np.nansum( cls*mw ) / np.nansum(mw)
        dme[key] = np.nansum( eng*mw ) / np.nansum(mw)
    return dmc, dme

def mask_dict_vol(dict_vol, mask=None):
    """
    Masks out values within the gradient value volumes
    which are not selected within the passed mask
    """
    if mask is not None:
        return dict_mul(dict_vol, mask)
    else:
        return dict_vol


def check_dict_nan( d ):
    for v in d.values():
        if np.any(np.isnan(v)):
            print "bad dict : ", d
            return False
    return True

def inter_save(pars, net, lc, grdts, malis_weights, wmsks, elapsed, it):
    # get file name
    filename, filename_current = znetio.get_net_fname( pars['train_save_net'], it )
    # save network
    znetio.save_network(net, filename )
    lc.save( pars, filename, elapsed )
    if pars['is_debug']:
        save_grdts(grdts, filename)
        if pars['is_malis'] and pars['is_stdio']:
            save_malis(malis_weights, filename)
        if pars['is_rebalance'] or pars['is_patch_rebalance']:
            save_rebalance(wmsks, filename)

    # Overwriting most current file with completely saved version
    shutil.copyfile(filename, filename_current)
