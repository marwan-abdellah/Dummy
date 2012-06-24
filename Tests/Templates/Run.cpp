/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File         : Volume
 * Author(s)    : Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created      : April 2011
 * Description  :
 * Note(s)      :
 *********************************************************************/

#include "ex_Templates.h"

int main()
{
	ex_Templates::streamOut((int) 1);
	ex_Templates::streamOut((float) 123.123456);
	ex_Templates::streamOut((long) 123456789);
	ex_Templates::streamOut((double) (0.123456789));

	return 0;
}
