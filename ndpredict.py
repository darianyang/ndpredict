"""
Use model built using model.py and feature array from calc_features.py.
Get new PDB file ASN residue N->D predicted probabilities.
"""

import pickle
import matplotlib.pyplot as plt

from model import NDPredict

# load pickel model object
ndp = pickle.load(open("ndp_model.pkl", 'rb'))
#print(ndp.feat_names)
# new ndp class to calc features
new_pdb = NDPredict()
new_pdb.proc_csv("data/1hk0_features.csv")
#print(new_pdb.feat_names)

# predict prob of new feats
pred = ndp.model.predict(new_pdb.X)
residues = [24, 33, 49, 118, 124, 137, 160]

pred_dict = dict(zip(residues, pred))
print(pred_dict)

plt.bar([str(i) for i in residues], pred)
plt.ylim(0,1)
plt.xlabel("$\gamma$D-Crystallin (1HK0) Residue Number")
plt.ylabel("Deamidation Probability")
plt.show()