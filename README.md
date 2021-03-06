# Sparse Localization using Particle Filters

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Your vehicle has been kidnapped and transported to a new location! Luckily it has a map of this location, a (noisy) 
GPS estimate of its initial location, and lots of (noisy) sensor and control data.

This project implements a 2 dimensional particle filter in C++. The particle filter will be given a map and some initial
localization information (analogous to what a GPS would provide). At each time step your filter will also get 
observation and control data.

![Screenshot](images/screenshot.png)

Note that despite what the name suggests, due to the way it is implemented, the particle filter won't work well (or at all) in situations 
where the vehicle is indeed kidnapped during the simulation. Speficially, particles are only resampled with replacement from the
previous generation; however, no _new_ particles will ever be created at random. This means that after convergence on a good pose
estimate, the filter would need to be reinitialized completely to handle a kidnapping situation. In this case, each generation's particle
weights should be low, so this situation could at least be detected.

This code makes use of the [libssrckdtree](https://www.savarese.com/software/libssrckdtree/) library
for aligning measurements and landmarks. The code is bundled in the `vendor/libssrckdtree` directory
and is licensed under an Apache 2.0 License.

## Running the Code
This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and install [uWebSocketIO](https://github.com/uNetworking/uWebSockets) 
for either Linux or Mac systems. For windows you can use either Docker, VMware, or even Windows 10 Bash on Ubuntu to
install uWebSocketIO.

Once the install for uWebSocketIO is complete, the main program can be built and ran by doing the following from the 
project top directory.

1. `mkdir build`
2. `cd build`
3. `cmake ..`
4. `make`
5. `./particle_filter`


If, for whatever reason, you do not want to build using libssrc, e.g. because it requires libraries not available on your system, you can enable "naive"
keypoint matching mode by running CMake as follows:

3. `cmake -DUSE_ASSOCIATION_NAIVE=On -DNUM_PARTICLES=100 ..`

Note that the number of particles will be reduced to improve performance.

Alternatively some scripts have been included to streamline this process, these can be leveraged by executing the following in the top directory of the project:

1. `./clean.sh`
2. `./build.sh`
3. `./run.sh`

Tips for setting up your environment can be found [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)

Here is the main protocol that main.cpp uses for uWebSocketIO in communicating with the simulator.

### Input values provided by the simulator to the c++ program

Sense noisy position data from the simulator:

* `["sense_x"]` <= X coordinate 
* `["sense_y"]` <= Y coordinate
* `["sense_theta"]` <= Orientation in radians

Previous velocity and yaw rate to predict the particle's transitioned state:

* `["previous_velocity"]` 
* `["previous_yawrate"]`

Receive noisy observation data from the simulator, in a respective list of x/y values

* `["sense_observations_x"]`
* `["sense_observations_y"]`


### Output values provided by the c++ program to the simulator

Best particle values used for calculating the error evaluation:

* `["best_particle_x"]`
* `["best_particle_y"]`
* `["best_particle_theta"]`

#### Optional message data used for debugging particle's sensing and associations:

For respective (x,y) sensed positions ID label:

* `["best_particle_associations"]`

For respective (x,y) sensed positions:

* `["best_particle_sense_x"]` <= list of sensed x positions
* `["best_particle_sense_y"]` <= list of sensed y positions


# Implementing the Particle Filter
The directory structure of this repository is as follows:

```
root
|   build.sh
|   clean.sh
|   CMakeLists.txt
|   README.md
|   run.sh
|
|___data
|   |   
|   |   map_data.txt
|   
|   
|___src
|   |   helper_functions.h
|   |   main.cpp
|   |   map.h
|   |   particle_filter.cpp
|   |   particle_filter.h
|   |   data_association_naive.cpp
|   |   data_association_libssrckdtree.cpp
|   
|   
|___vendor
    |___libssrckdtree
        | ...
    
```

The main file of interest `particle_filter.cpp` in the `src` directory. 
The file contains the scaffolding of a `ParticleFilter` class and some associated methods. Different methods of
associating landmarks with measurements are implemented; these can be found in the `data_association_naive.cpp` and
`data_association_libssrckdtree.cpp` files.

## Inputs to the Particle Filter
You can find the inputs to the particle filter in the `data` directory.

#### The Map*
`map_data.txt` includes the position of landmarks (in meters) on an arbitrary Cartesian coordinate system. 
Each row has three columns:

1. x position
2. y position
3. landmark id

### All other data the simulator provides, such as observations and controls.

> * Map data provided by 3D Mapping Solutions GmbH.

## Success Criteria
If your particle filter passes the current grading code in the simulator (you can make sure you have the current
version at any time by doing a `git pull`), then you should pass!

The things the grading code is looking for are:

1. **Accuracy**: your particle filter should localize vehicle position and yaw to within the values specified in the parameters `max_translation_error` and `max_yaw_error` in `src/main.cpp`.
2. **Performance**: your particle filter should complete execution within the time of 100 seconds.
