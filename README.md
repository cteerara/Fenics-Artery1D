
# 1D arterial flow 
This code solves the 1D arterial flow in bifurcating compliant vessel using FEniCS finite element library. The background mathematics and implementation is detailed in Implementation_Details.pdf.

# Demo 
To run the demo, go to src/ and run
```
python3 demo.py inputs/intputFile
```
where several inputFiles are provided in src/inputs*

The program outputs 3 files: `<tag>_A.npy`, `<tag>_Q.npy`, and `<tag>_properties.pkl` where `<tag>` is the name of the input file. These are all under directory `<tag>/` directory.  The `*.npy` files are 3 dimensional array with shape `(numTimeStep, numNodes, numVessels)` and the `*.pkl` file is a dictionary containing vessel properties. 

# Post processing
Example on how to post process these files can be found in `src/plotWave.py`. This file is written to post process cases with 3 vessel segments. This includes `prosthesis.in`, `noProsthesis.in`, and `yBifurcation.in` in the demo. This code plots pressure wave propagation over time and pressure at the center of each vessel segment over time. To run this code do:
```
python3 plotWave.py <tag>/
```

# Input file structure
The input file defines vessel connectivity and the properties on each vessel segment. The input file is separated into two parts in the same input file. Note that lines starting with `#` are ignored.
## 1. Vessel connectivity
This part defines vessel connectivity as a tree structure with the vessel receiving the inlet boundary condition (source vessel) is the root node. This input starts with the keyword `Connectivity`. Input file format is as follows:
1. The format is `Connectivity : p : d1,d2,d3,...,dn` where `p` is the ID of the parent vessel, and `d1,d2,...,dn` are the ids of the daughter vessels.
2. Vessel ID must start from 0 and the ID cannot be skipped, e.g., 0,1,2,3 is OK while 0,1,2,4 is NOT OK.
3. Vessel ID 0 must be present and is considered the "source" vessel where your inlet boundary condition will be applied.
4. parent vessels must have ID lower than all the daughter vessels
5. If the vessel does not have daughter, you can omit it from the `Connectivity` input

For example, a complex network network:
```
                |---4---
         |---1--- 
         |      |---5---
         |
 ---0-------2---|---6---
         |
         ---3---|---7---
```
has the following connectivity format:
```
Connectivity : 0 : 1,2,3 
Connectivity : 1 : 4,5
Connectivity : 2 : 6
Connectivity : 3 : 7
```

## 1. Vessel properties
The input file also specify input properties. Vessel properties starts with the keyword `Value`. The format is as follows:
1. `Value : <var> = <var_0> , <var_1> , ... <var_N>` where `<var>` is the name of the variable, and `<var_i>` is the value of that variable on segment `i`. Each `<var>` entreis are separated by a `,`
2. Due to the nature of how the problem is solve, vessel properties on each segments do not need to be continuous. See `Implementation_Detaion.pdf` for more info.

For the above network, one can define the above vessel network follows
```
# L is the length of the domain
# ne is the number of elements
# r0 is the vessel's initial radius
# E is the vessel's elastic modulus
# h0 is the vessel's wall thickness
# degA and degQ are the interpolating polynomial degree

# Arteries     0    , 1    , 2    , 3    , 4    , 5    , 6    , 7  
Value : L    = 15   , 15   , 15   , 15   , 15   , 15   , 15   , 15   
Value : ne   = 128  , 128  , 128  , 128  , 128  , 128  , 128  , 128 
Value : r0   = 0.5  , 0.5  , 0.5  , 0.5  , 0.5  , 0.5  , 0.5  , 0.5 
Value : Q0   = 0    , 0    , 0    , 0    , 0    , 0    , 0    , 0  
Value : E    = 3e6  , 3e6  , 3e6  , 3e6  , 3e6  , 3e6  , 30e6 , 3e6
Value : h0   = 0.05 , 0.05 , 0.05 , 0.05 , 0.05 , 0.05 , 0.05 , 0.05
Value : degA = 1    , 1    , 1    , 1    , 1    , 1    , 1    , 1  
Value : degQ = 1    , 1    , 1    , 1    , 1    , 1    , 1    , 1 
```

# Editing inlet BC and fluid properties
Currently, inlet boundary condition and fluid properties are hard-coded in `src/demo.py`. One can play around with different values there.





