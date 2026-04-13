#this file is part of litwin-kumar_doiron_formation_2014
#Copyright (C) 2014 Ashok Litwin-Kumar
#see README for more information

#this is how to save compressed connectivity data in the HDF5 format
fid = h5open("data.h5","w")

g = create_group(fid, "data")

g["popmembers"] = popmembers
g["weights"] = weights

close(fid)
