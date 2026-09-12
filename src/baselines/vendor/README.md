# Official PPI model sources

- TUnA: https://github.com/Wang-lab-UCSD/TUnA, commit b5bda8fee261a4f27821738db995cf5883dcd133; results/bernett/TUnA/model.py (model classes) and lookahead.py. MIT license retained. Removed standalone trainer/tester and their unused utility imports; attention scale is a nonpersistent device-moving buffer. The published block-diagonal interaction mask and readout are unchanged.
- TUnA uncertainty head: https://github.com/jlparkI/uncertaintyAwareDeepLearn, tag 0.0.5, commit 18565eb86026800817857e37243ae81f15f089d7; classic_rffs.py unchanged, MIT license retained.
- PPITrans: https://github.com/LtECoD/PPITrans, commit 0e55e91e510596b33004ee549be0ce5492f2f906; module/{encoder,decoder,utils}.py. Replace the Fairseq model base with nn.Module and localize utility imports; all encoder/decoder math is unchanged. No upstream LICENSE file was present at this revision; no license is inferred or assigned here.

Adapters, benchmark training and tests live outside this directory. Frozen token input dimension is 1536, maximum length 1024 for both models. No upstream supervised PPI checkpoint is used.

Trailing whitespace is normalized in the adapted TUnA model and PPITrans encoder files.
