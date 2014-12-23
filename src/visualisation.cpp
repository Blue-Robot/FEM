// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <FreeImage.h>


// C/C++ Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <rendercheck_gl.h>

#include <curand.h>

#include "visualisation.h"
#include "FEM_common.h"

using namespace OpenMesh;

int vertices;
int faces;
uint *faceVertices;
double2 *d_fn;
int frame_counter = 1;
bool save = true;

GLuint dataVbo;
GLuint indexVbo;
struct cudaGraphicsResource *vbo_res[2];
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

bool initGL(int *argc, char **argv);
void keyboard(unsigned char key, int /*x*/, int /*y*/);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, int size, int mode, uint flag);


extern "C" bool initialize(int *argc, char **argv, int v, int f) {
	if (!initGL(argc, argv))
		return false;

	vertices = v;
	faces = f;

	return true;
}

extern "C" void set_fn(double2 *dev_fn) {
	d_fn = dev_fn;
}

extern "C" void display() {
	format(d_fn, &vbo_res[0], vertices);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y+45, 0.0, 1.0, 0.0);
	// bind buffer
	glBindBuffer(GL_ARRAY_BUFFER, dataVbo);

	// render from the vbo
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glColorPointer(4, GL_DOUBLE, 0, (const GLvoid *) (sizeof(GLdouble) * vertices * 4));
	glNormalPointer(GL_FLOAT, 0, (const GLvoid *) (sizeof(GLfloat) * vertices * 4 + sizeof(GLfloat) * vertices * 4));

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);

	glDrawElements(GL_TRIANGLES, faces * 3, GL_UNSIGNED_INT, 0);

	// unbind buffer
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);

	glutSwapBuffers();
	if (save) {
		int width = 1000;
		int height = 1000;

		std::stringstream ss;
		ss << "/home/clood/cuda-workspace/FEM/Video/frame_" << std::setw(10) << std::setfill('0') << frame_counter << ".bmp";

		BYTE* pixels = new BYTE[ 3 * width * height];

		glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, pixels);

		FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, width, height, 3 * width, 24, 0x0000FF, 0xFF0000, 0x00FF00, false);
		FreeImage_Save(FIF_BMP, image, ss.str().c_str(), 0);

		FreeImage_Unload(image);
		delete [] pixels;
	}
	frame_counter++;
}

bool initGL(int *argc, char **argv) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(1000, 1000);
	glutInitWindowPosition((glutGet(GLUT_SCREEN_WIDTH) - 1000) / 2, (glutGet(GLUT_SCREEN_HEIGHT) - 1000) / 2);
	glutCreateWindow("FEM");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	glutTimerFunc(10, timerEvent, 0);

	// initialize necessary OpenGL extensions
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "\nERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	// Setup our viewport and viewing modes
	glViewport(0, 0, 1000, 1000);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat) 1000 / (GLfloat) 1000, 0.1, 10.0);

	SDK_CHECK_ERROR_GL();

	return true;
}

void keyboard(unsigned char key, int /*x*/, int /*y*/) {
	switch (key) {
	case (27):
		exit(EXIT_SUCCESS);
		break;
	}
}

void mouse(int button, int state, int x, int y) {
	if (state == GLUT_DOWN) {
		mouse_buttons |= 1 << button;
	} else if (state == GLUT_UP) {
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void motion(int x, int y) {
	float dx, dy;
	dx = (float) (x - mouse_old_x);
	dy = (float) (y - mouse_old_y);

	if (mouse_buttons & 1) {
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	} else if (mouse_buttons & 4) {
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void timerEvent(int value) {
	glutPostRedisplay();
	glutTimerFunc(10, timerEvent, 0);
}

void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, int size, int mode, uint flag) {

	glGenBuffers(1, vbo);
	glBindBuffer(mode, *vbo);
	glBufferData(mode, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(mode, 0);

	// Register VBO with Cuda
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, flag));
}

extern "C" void initiateVBOData(SimpleTriMesh mesh) {
	createVBO(&dataVbo, &vbo_res[0], vertices * 4 * 3 * sizeof(GLdouble), GL_ARRAY_BUFFER, cudaGraphicsRegisterFlagsNone);
	createVBO(&indexVbo, &vbo_res[1], faces * 3 * sizeof(GLuint), GL_ELEMENT_ARRAY_BUFFER, cudaGraphicsRegisterFlagsNone);

	GLfloat position[vertices * 4];
	GLfloat normal[vertices * 4];

	for (int i = 0; i < vertices; i++) {
		SimpleTriMesh::VertexHandle v = mesh.vertex_handle(i);
		OpenMesh::Vec3d p = mesh.point(v);
		OpenMesh::Vec3d n = mesh.normal(v);
		for (int j = 0; j < 3; j++) {
			position[i * 4 + j] = p.values_[j];
			normal[i * 4 + j] = n.values_[j];
		}
		position[i * 4 + 3] = 1.0;
	}

	faceVertices = new uint[faces*3];
	for (int i = 0; i < faces; i++) {
		FaceHandle f = mesh.face_handle(i);

		int counter = 0;
		for(SimpleTriMesh::FaceVertexIter fvIter = mesh.fv_begin(f); fvIter != mesh.fv_end(f); ++fvIter) {
			VertexHandle v = *fvIter;
			faceVertices[i*3 + counter] = v.idx();
			counter++;
		}
	}

	glBindBuffer(GL_ARRAY_BUFFER, dataVbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexVbo);

	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(position), position);
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(position) + sizeof(GLfloat) * vertices * 4, sizeof(normal), normal);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(faceVertices) * faces * 3, faceVertices, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}
