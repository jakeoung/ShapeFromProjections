import os
import time
import argparse
import datetime

#try:
#    fmain = os.path.basename(__file__)
#except:
#    fmain = 'jupyter'

parser = argparse.ArgumentParser(add_help=False, fromfile_prefix_chars='@')
parser.add_argument('-data', type=str, default='2bunny', help='folder naming: data/[nmaterials]dataset')
parser.add_argument('-niter', type=int, default=500, help='number of iterations')
parser.add_argument('-niter0', type=int, default=0, help='iter before refinement')
parser.add_argument('-wlap', type=float, default=10., help='reg. parameter')
parser.add_argument('-eta', type=float, default=0, help='noise level')

parser.add_argument('-wedge', type=float, default=2., help='reg. parameter')
parser.add_argument('-wflat', type=float, default=0.01, help='flatten loss parameter')
parser.add_argument('-b', type=int, default=0, help='batch size')
parser.add_argument('-lr', type=float, default=0.01, help='step size')
parser.add_argument('-nmu0', type=int, default=1, help='number of fixed mus. 1 means we dont optimize the background attenuation')
parser.add_argument('-subdiv', type=int, default=3, help='number of fixed mus. 1 means we dont optimize the background attenuation')
    
#parser.add_argument('-eta', type=float, default=0., help='noise std on the sinogram data')
#parser.add_argument('-K', type=int, default=30, help='number of projection angles')

args_key, unparsed = parser.parse_known_args()

def get_fresult(dic_):
    #fresult_ = time.strftime('%m%d_')
    fresult_ = ''
    key_params = dic_.keys()
    for key in sorted(key_params):
        value_str = str(dic_[key])
        if value_str.find('_') >= 0:
            value_str = value_str.replace("_", "-")

        if key == 'data':
            continue
            
        fresult_ += '-' + str(key) + '_' + value_str + '_'
    return fresult_

fresult = get_fresult(vars(args_key))

# args.dataroot = '../data/'
# args.resroot = '../result/'
####################################
## Parsing secondary arguments
####################################
parser2 = argparse.ArgumentParser(parents=[parser],  fromfile_prefix_chars='@')

#parser2.add_argument('-init', type=str, default='gtv', help='hist', choices=['a'])


parser2.add_argument('-verbose', '-v', type=int, default=1, help='control verbose mode')
parser2.add_argument('-cuda', type=int, default=-1, help='the number of GPU device')
parser2.add_argument('-dataroot', type=str, default='../data/', help='data root')

parser2.add_argument('-resroot', type=str, default='../result', help='result root')
# parser2.add_argument('-init', type=str, default='', help='initialization file of the curve. (npy)')

args, unparsed = parser2.parse_known_args(namespace=args_key) # for jupyter

if args.cuda >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    
#args = parser2.parse_args(namespace=args_key) 

#####################################
### Directory structure example:
### args.dataroot: .../useg/data/dataset-name/
### args.dresult : .../useg/result/dataset-name/fckpt
###
### - .../useg/data/dataset-name/train/img1.png
### - .../useg/result/dataset-name/.../iter/...
#####################################

def update_args(args, make_dir=True):
    # for jupyter lab
    m = int(args.data[0])
    try:
        if m >= 2 and m <= 9:
            args.nmaterials = m
        elif m == 1:
            args.nmaterials = int(args.data[0:2])
    
    except:
        print("args.nmaterials can't be parsed and is set as 2")
        args.nmaterials = 2
    
    if make_dir == False:
        return
        
    # datetime.date(2010, 6, 16).isocalendar()[1]
    # iso = datetime.datetime.utcnow().isocalendar()
    # iso[1]:02d
    
    args.ddata = os.path.join(args.dataroot, args.data) + '/'
    args.dresult = os.path.join(args.resroot, args.data) + '/ours_' + fresult + '/'
    os.makedirs(args.dresult, exist_ok=True)
    
    print(args)

update_args(args, make_dir=False)


