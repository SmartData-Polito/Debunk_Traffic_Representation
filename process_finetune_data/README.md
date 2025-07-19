# Data Process
##  Data Split
- Per-packet split: Mix all packets and split by 8:1:1
- Per-flow split: Split the data based on disjoint flows (TODO: optimize the code performance on *pkt_classification*)

## Data Process for other models 
*Shallow ML, ET-BERT, YaTC, PCAP Encoder, NetMamba, TrafficFormer and netFound*

- Added the main method of processing data packets, based on the processing flow of the original algorithms
- The data processing way of YaTC and NetMamba are same
- For NetMamba, netFound, and YaTC, adopted the padding way that repeats the same packet up to the input limitation.
- For TrafficFormer, used data augmentation with 5 times.
- For Shallow ML, retreived all related fields and normalized some fields data. (TODO: optimize the code)