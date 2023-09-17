/*

*/
SetFactory("OpenCASCADE");
size = 0.5;
// Points
Point(1) = {0, 1, 1, size};
Point(2) = {0, -1, 1, size};
Point(3) = {0, -1, -1, size};
Point(4) = {0, 1, -1, size};


// Lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {4, 1, 2, 3};
Plane Surface(1) = {1};
Transfinite Surface {1};
Recombine Surface {1};

// Volume
Extrude {5, 0, 0} {
  Surface{1}; Layers {30}; Recombine;
}


//+
Physical Surface(13) = {1};
//+
Physical Curve(16) = {7};
//+
Physical Volume(15) = {1};


//+
Transfinite Volume{1};
//+
Transfinite Surface {6};
//+
Transfinite Curve {9, 11, 12, 7} = 11 Using Progression 1;
//+
Transfinite Surface {1};
//+
Transfinite Curve {1, 2, 3, 4} = 11 Using Progression 1;

