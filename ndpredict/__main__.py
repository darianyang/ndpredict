"""
Use model built using model.py and feature array from calc_features.py.
Get new PDB file ASN residue N->D predicted probabilities.
"""

import pickle
import matplotlib.pyplot as plt

from ndpredict.model import NDPredict
from ndpredict.command_line import *
from ndpredict.calc_features import *

def main():

    """
    Command line
    """
    # Create command line arguments with argparse
    argument_parser = create_cmd_arguments()
    # Retrieve list of args
    args = handle_command_line(argument_parser)

    """
    NDPredict
    """
    # TODO: add pdb4amber/pdbfixer step to pre-process input PDB
    #       also make basic argparser

    # load pickel model object
    #ndp = pickle.load(open("ndp_model.pkl", 'rb'))
    #ndp = pickle.load(open(args.model, 'rb'))
    #print(ndp.feat_names)

    # calc features of input pdb
    cf = Calc_Features(args.pdb, asns=None, chainid=args.chainid)
    fa = cf.construct_feat_array()
    fa.to_csv(f"{args.pdb[:-4]}_features.csv", index=False)

    # new ndp class to calc features
    new_pdb = NDPredict()
    # load pre-trained model into new NDPredict instance
    new_pdb.model = pickle.load(open(args.model, 'rb'))
    #new_pdb.proc_csv("data/1hk0_features.csv")
    #new_pdb.proc_csv("data/1gb1_features.csv")
    #new_pdb.proc_csv("data/2m3t_features.csv")

    # process updated feat data
    new_pdb.proc_csv(f"{args.pdb[:-4]}_features.csv")

    # predict prob of new feats
    pred = new_pdb.model.predict(new_pdb.X)

    # 1hk0
    # residues = [24, 33, 49, 118, 124, 137, 160]
    # # 1gb1
    # residues = [8, 35, 37]
    # # 2m3t
    # residues = [15, 38, 54, 77, 144]
    # I can grab asn residues directly from Calc_Features object
    # but convert from res index (0 index) to res id (1 index)
    residues = np.add(cf.asns, 1)

    # TODO: send to logger
    pred_dict = dict(zip(residues, pred))
    print(pred_dict)

    # TODO: oop based mpl plotting?
    plt.bar([str(i) for i in residues], pred)
    plt.ylim(0,1)
    #plt.xlabel("$\gamma$D-Crystallin (1HK0) Residue Number")
    #plt.xlabel("GB1 Residue Number")
    #plt.xlabel("$\gamma$S-Crystallin (2M3T) Residue Number")
    #plt.ylabel("Deamidation Probability")
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.title(args.title)
    plt.show()

if __name__ == "__main__": 
    main()