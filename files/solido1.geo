// Gmsh project created on Thu Jun 11 12:24:03 2020


rad = 1.0;
size = 1.0;

// Points
Point(1) = {0, 0, 0, size};
Point(2) = {rad, 0, 0, size};
Point(3) = {rad, rad, 0, size};
Point(4) = {0, rad, 0, size};

// Lines
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};


//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Surface(1) = {1};
//+
Physical Surface(5) = {1};
