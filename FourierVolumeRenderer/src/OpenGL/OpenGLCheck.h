#ifndef OPENGLCHECK_H
#define OPENGLCHECK_H

#include <X11/Xlib.h>

/* @ */
bool isOpenGLAvailable()
{
    /* @ X11 display */
    Display *eXDisplay = XOpenDisplay(NULL);

    /* @ If null, NO OpenGL will be found */
    if (eXDisplay == NULL)
        return false;
    else
    {
        XCloseDisplay(eXDisplay);
        return true;
    }
}


#endif // OPENGLCHECK_H
