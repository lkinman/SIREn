'''
Create a MIXED star file from a particle stack and ctf parameters
Adapted by Maria Carreira from cryoDRGN write_starfile.py 

'''

import argparse
import numpy as np
import sys, os
import pickle
import pandas as pd

from cryodrgn import dataset
from cryodrgn import utils
from cryodrgn import starfile
from cryodrgn import mrc
log = utils.log 

HEADERS = ['_rlnImageName',
           '_rlnDefocusU',
           '_rlnDefocusV',
           '_rlnDefocusAngle',
           '_rlnVoltage',
           '_rlnSphericalAberration',
           '_rlnAmplitudeContrast',
           '_rlnPhaseShift']

POSE_HDRS = ['_rlnAngleRot',
             '_rlnAngleTilt',
             '_rlnAnglePsi',
             '_rlnOriginX',
             '_rlnOriginY']

MICROGRAPH_HDRS = ['_rlnMicrographName',
                   '_rlnCoordinateX',
                   '_rlnCoordinateY']

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--particles1', help='Input particles (.mrcs, .txt, .star, .cs)')
    parser.add_argument('--ctf1', help='Input ctf.pkl')
    parser.add_argument('--poses1', help='Optionally include pose.pkl') 
    parser.add_argument('--particles2', help='Input particles (.mrcs, .txt, .star, .cs)')
    parser.add_argument('--ctf2', help='Input ctf.pkl')
    parser.add_argument('--poses2', help='Optionally include pose.pkl') 
    parser.add_argument('--freq1', required=True, type=float, help='Frequency of First star file')
    parser.add_argument('--freq2', required=True, type=float, help='Frequency of Second star file')
    parser.add_argument('--ind1', help='Optionally filter by selected index array (.pkl)')
    parser.add_argument('--ind2', help='Optionally filter by selected index array (.pkl)')
    parser.add_argument('--datadir', type=os.path.abspath, help='Path prefix to particle stack if loading relative paths from a .star or .cs file')
    parser.add_argument('--full-path', action='store_true', help='Write the full path to particles (default: relative paths)')
    parser.add_argument('-o', type=os.path.abspath, required=True, help='Output .star file')

    group = parser.add_argument_group('Optionally include additional star file columns')
    group.add_argument('--ref-star1', help='Reference star file from original import')
    group.add_argument('--ref-star-relion311', action='store_true', help='Flag for relion3.1 star format')
    group.add_argument('--ref-star2', help='Reference star file from original import')
    group.add_argument('--ref-star-relion312', action='store_true', help='Flag for relion3.1 star format')
    group.add_argument('--keep-micrograph1', action='store_true', help='Include micrograph coordinate headers')
    group.add_argument('--keep-micrograph2', action='store_true', help='Include micrograph coordinate headers')
    return parser

def main(args):
    assert args.o.endswith('.star')
    particles1 = dataset.load_particles(args.particles1, lazy=True, datadir=args.datadir)
    particles2 = dataset.load_particles(args.particles2, lazy=True, datadir=args.datadir)
    ctf1 = utils.load_pkl(args.ctf1)
    ctf2 = utils.load_pkl(args.ctf2)
    assert ctf1.shape[1] == 9, "Incorrect CTF pkl format"
    assert ctf2.shape[1] == 9, "Incorrect CTF pkl format"
    assert len(particles1) == len(ctf1), f"{len(particles1)} != {len(ctf1)}, Number of particles != number of CTF paraameters"
    assert len(particles2) == len(ctf2), f"{len(particles2)} != {len(ctf2)}, Number of particles != number of CTF paraameters"
    if args.poses1:
        poses1 = utils.load_pkl(args.poses1)
        assert len(particles1) == len(poses1[0]), f"{len(particles1)} != {len(poses1)}, Number of particles != number of poses"
    log('{} particles'.format(len(particles1)))

    if args.poses2:
        poses2 = utils.load_pkl(args.poses2)
        assert len(particles2) == len(poses2[0]), f"{len(particles2)} != {len(poses2)}, Number of particles != number of poses"
    log('{} particles'.format(len(particles2)))
    
    if args.ref_star1:
        ref_star1 = starfile.Starfile.load(args.ref_star1, relion311=args.ref_star_relion311)
        assert len(ref_star1) == len(particles1), f"Particles in {args.particles1} must match {args.ref_star1}"

    if args.ref_star2:
        ref_star2 = starfile.Starfile.load(args.ref_star2, relion312=args.ref_star_relion312)
        assert len(ref_star2) == len(particles2), f"Particles in {args.particles2} must match {args.ref_star2}"
    

    if args.ind1:
        ind1 = utils.load_pkl(args.ind1)
        log(f'Filtering to {len(ind1)} particles')
        particles1 = [particles1[ii] for ii in ind1]
        ctf1 = ctf1[ind1]
        if args.poses1: 
            poses1 = (poses1[0][ind1], poses1[1][ind1])
        if args.ref_star:
            ref_star.df = ref_star.df.loc[ind1] 
    else:
        ind1 = np.arange(len(particles1))
    
    ind1 += 1 

    if args.ind2:
        ind2 = utils.load_pkl(args.ind2)
        log(f'Filtering to {len(ind2)} particles')
        particles2 = [particles2[ii] for ii in ind2]
        ctf2 = ctf2[ind2]
        if args.poses2: 
            poses2 = (poses2[0][ind2], poses2[1][ind2])
        if args.ref_star:
            ref_star.df = ref_star.df.loc[ind2] 
    else:
        ind2 = np.arange(len(particles2))

    ind2 += 1 
    
    image_names1 = [img.fname for img in particles1]
    if args.full_path:
        image_names1 = [os.path.abspath(img.fname) for img in particles1]
    names1 = [f'{i}@{name}' for i,name in zip(ind1, image_names1)]
    
    ctf1 = ctf1[:,2:]
    
    image_names2 = [img.fname for img in particles2]
    if args.full_path:
        image_names2 = [os.path.abspath(img.fname) for img in particles2]
    names2 = [f'{i}@{name}' for i,name in zip(ind2, image_names2)]
    
    ctf2 = ctf2[:,2:]
    
    # convert poses
    if args.poses1:
        eulers1 = utils.R_to_relion_scipy(poses1[0]) 
        D1 = particles1[0].get().shape[0]
        trans1 = poses1[1] * D1 # convert from fraction to pixels
        
    if args.poses2:
        eulers2 = utils.R_to_relion_scipy(poses2[0]) 
        D2= particles2[0].get().shape[0]
        trans2= poses2[1] * D2 # convert from fraction to pixels

        
    data1 = {HEADERS[0]:names1}
    for i in range(7):
        data1[HEADERS[i+1]] = ctf1[:,i]
    if args.poses1:
        for i in range(3):
            data1[POSE_HDRS[i]] = eulers1[:,i]
        for i in range(2):
            data1[POSE_HDRS[3+i]] = trans1[:,i]
    df1 = pd.DataFrame(data=data1)
    headers = HEADERS + POSE_HDRS if args.poses1 else HEADERS   
    if args.keep_micrograph1:
        assert args.ref_star1, "Must provide reference .star file with micrograph coordinates"
        log(f'Copying micrograph coordinates from {args.ref_star1}')
        for h in MICROGRAPH_HDRS:
            df1[h] = ref_star1.df1[h]
        headers += MICROGRAPH_HDRS
        
    
    data2 = {HEADERS[0]:names2}
    for j in range(7):
        data2[HEADERS[j+1]] = ctf2[:,j]
    if args.poses2:
        for j in range(3):
            data2[POSE_HDRS[j]] = eulers2[:,j]
        for j in range(2):
            data2[POSE_HDRS[3+j]] = trans2[:,j]
    df2 = pd.DataFrame(data=data2)
    if args.keep_micrograph2:
        assert args.ref_star2, "Must provide reference .star file with micrograph coordinates"
        log(f'Copying micrograph coordinates from {args.ref_star2}')
        for h in MICROGRAPH_HDRS:
            df2[h] = ref_star2.df2[h]
        headers += MICROGRAPH_HDRS

    
    #shuffle the two resulting star files
    freq1=args.freq1
    freq2=args.freq2
    
    def random_choice(df1, df2, freq1, freq2):
        output1=df1.loc[np.random.choice(range(0, len(df1)), size = int(freq1), replace = False)]
        output2=df2.loc[np.random.choice(range(0, len(df2)), size = int(freq2), replace = False)]
        df=pd.concat([output1, output2], axis=0)
        return df.sample(frac=1).reset_index(drop=True)
        
    output=random_choice(df1, df2, freq1, freq2)
    s = starfile.Starfile(headers, output)
    s.write(args.o)

if __name__ == '__main__':
    main(parse_args().parse_args())
