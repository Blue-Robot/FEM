#include "common.h"

extern "C" void display();

extern "C" bool initialize(int *argc, char **argv, int v, int f);

extern "C" void set_fn(double2 *dev_fn);

extern "C" void initiateVBOData(SimpleTriMesh mesh);
