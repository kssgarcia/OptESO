// Points
Point(1) = {0, 0, 0, size};
Point(2) = {10, 0, 0, size};
Point(3) = {10, 1, 0, size};
Point(4) = {0, 1, 0, size};

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
//+
Physical Surface(6) = {1};
