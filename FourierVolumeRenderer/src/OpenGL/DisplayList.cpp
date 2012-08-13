#include "DisplayList.h"


GLuint OpenGL::setDisplayList(float center, float sideLength)
{
    printf("Creating Display List ... \n");

    // Center (Z = 0), Side Lenght = 1, Just 1 Slice, 4 Vertices, 3 Coordinates

    int numSlices           = 1;
    int numElements         = 4 * numSlices;
    GLfloat *vertexList	= new GLfloat [3 * numElements];

    // Fill the Display List with Vertecies
    vertexList[0] = -sideLength / 2;
    vertexList[1] = -sideLength / 2;
    vertexList[2] =  center;

    vertexList[3] =  sideLength / 2;
    vertexList[4] = -sideLength / 2;
    vertexList[5] =  center;

    vertexList[6] =  sideLength / 2;
    vertexList[7] =  sideLength / 2;
    vertexList[8] =  center;

    vertexList[9] = -sideLength / 2;
    vertexList[10] =  sideLength / 2;
    vertexList[11] =  center;

    // Fill the Display List with Vertecies
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, vertexList);

    // Create the disply list
    GLuint displyList = glGenLists(1);
    glNewList(displyList, GL_COMPILE);
    glDrawArrays(GL_QUADS, 0, numElements);
    glEndList();

    delete [] vertexList;

    printf("	Display List Created Successfully \n\n");

    return displyList;
}

