#include "DisplayList.h"
#include "Utilities/MACROS.h"
#include "Utilities/Utils.h"


GLuint OpenGL::setDisplayList(float iCenter, float iSideLength)
{
    INFO("Creating DisplayList")

    /* @ How thw display list is being created
     *   iCenter (Z = 0), Side Lenght = 1,
     *   Just 1 Slice, 4 Vertices, 3 Coordinates
     */

    int numSlices           = 1;
    int numElements         = 4 * numSlices;
    GLfloat *vertexList	= (GLfloat*) malloc (3 * numElements * sizeof(GLfloat));

    /* @ Fill the DisplayList with vertecies */
    vertexList[0] = -iSideLength / 2;
    vertexList[1] = -iSideLength / 2;
    vertexList[2] =  iCenter;

    vertexList[3] =  iSideLength / 2;
    vertexList[4] = -iSideLength / 2;
    vertexList[5] =  iCenter;

    vertexList[6] =  iSideLength / 2;
    vertexList[7] =  iSideLength / 2;
    vertexList[8] =  iCenter;

    vertexList[9] = -iSideLength / 2;
    vertexList[10] =  iSideLength / 2;
    vertexList[11] =  iCenter;

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, vertexList);

    /* @ Create the final DisplayList */
    GLuint displyList = glGenLists(1);
    glNewList(displyList, GL_COMPILE);
    glDrawArrays(GL_QUADS, 0, numElements);
    glEndList();

    free(vertexList);

    INFO("Creating DisplayList DONE ");

    return displyList;
}

