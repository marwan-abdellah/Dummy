/*
 * SliceExtraction.cpp
 *
 *  Created on: May 23, 2012
 *      Author: abdellah
 */

#include "SliceExtraction.h"

void init

void OpenGL::Slice::extractSlice(float sCenter, GLuint* bufferID, GLuint* textureID)
{
	printf("Extracting Fourier Slice ... \n");

	SetDisplayList(sCenter);

	// Render to Framebuffer Render Target
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, mFBO_ID);

	// Clear Buffer
	glClear(GL_COLOR_BUFFER_BIT);

	// Enable 3D Texturing
	glEnable(GL_TEXTURE_3D);

	// Setup Texture Variables & Parameters
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	// Bind 3D Texture
	glBindTexture(GL_TEXTURE_3D, mVolTexureID);

	// Adjust OpenGL View Port
	glViewport(-128,-128,512,512);

	// Texture Corrdinate Generation
	glEnable(GL_TEXTURE_GEN_S);
	glEnable(GL_TEXTURE_GEN_T);
	glEnable(GL_TEXTURE_GEN_R);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// 6 Main - Clip Planes
	static GLdouble eqx0[4] = { 1.0, 0.0, 0.0, 0.0};
	static GLdouble eqx1[4] = {-1.0, 0.0, 0.0, 1.0};
	static GLdouble eqy0[4] = {0.0,  1.0, 0.0, 0.0};
	static GLdouble eqy1[4] = {0.0, -1.0, 0.0, 1.0};
	static GLdouble eqz0[4] = {0.0, 0.0,  1.0, 0.0};
	static GLdouble eqz1[4] = {0.0, 0.0, -1.0, 1.0};

	// Define Equations for Automatic Texture Coordinate Generation
	static GLfloat x[] = {1.0, 0.0, 0.0, 0.0};
	static GLfloat y[] = {0.0, 1.0, 0.0, 0.0};
	static GLfloat z[] = {0.0, 0.0, 1.0, 0.0};

	glPushMatrix ();

	// Transform (Rotation Only) the Viewing Direction
	/*
	 * We don't need except the - 0.5 translation in each dimension to adjust
	 * the texture in the center of the scene
	 */
	glRotatef(-mXrot, 0.0, 0.0, 1.0);
	glRotatef(-mYrot, 0.0, 1.0, 0.0);
	glRotatef(-mZrot, 1.0, 0.0, 0.0);
	glTranslatef(-0.5, -0.5, -0.5);

	// Automatic Texture Coord Generation.
	glTexGenfv(GL_S, GL_EYE_PLANE, x);
	glTexGenfv(GL_T, GL_EYE_PLANE, y);
	glTexGenfv(GL_R, GL_EYE_PLANE, z);

	// Define the 6 Basic Clipping Planes (of the UNITY CUBE)
	glClipPlane(GL_CLIP_PLANE0, eqx0);
	glClipPlane(GL_CLIP_PLANE1, eqx1);
	glClipPlane(GL_CLIP_PLANE2, eqy0);
	glClipPlane(GL_CLIP_PLANE3, eqy1);
	glClipPlane(GL_CLIP_PLANE4, eqz0);
	glClipPlane(GL_CLIP_PLANE5, eqz1);

	glPopMatrix ();

	// Enable Clip Planes
	glEnable(GL_CLIP_PLANE0);
	glEnable(GL_CLIP_PLANE1);

	glEnable(GL_CLIP_PLANE2);
	glEnable(GL_CLIP_PLANE3);

	glEnable(GL_CLIP_PLANE4);
	glEnable(GL_CLIP_PLANE5);

	// Render Enclosing Rectangle (Only at (0,0) Plane)
	glCallList(mDiaplayList);
	glPopMatrix();

	// Disable Texturing
	glDisable(GL_TEXTURE_3D);
	glDisable(GL_TEXTURE_GEN_S);
	glDisable(GL_TEXTURE_GEN_T);
	glDisable(GL_TEXTURE_GEN_R);
	glDisable(GL_CLIP_PLANE0);
	glDisable(GL_CLIP_PLANE1);
	glDisable(GL_CLIP_PLANE2);
	glDisable(GL_CLIP_PLANE3);
	glDisable(GL_CLIP_PLANE4);
	glDisable(GL_CLIP_PLANE5);

	// Unbind the Framebuffer
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

	// Render Using the Texture Target
	glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

	// Enable 2D Texturing
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, *textureID);

	glPushMatrix();
	glLoadIdentity();
	glBegin(GL_QUADS);
		glNormal3f(0.0f, 0.0f, 1.0);
		glTexCoord2f(0.0,0.0);		glVertex3f(0.0,0.0,0.0);
		glTexCoord2f(1.0,0.0);		glVertex3f(1.0,0.0,0.0);
		glTexCoord2f(1.0,1.0);		glVertex3f(1.0,1.0,0.0);
		glTexCoord2f(0.0,1.0);		glVertex3f(0.0,1.0,0.0);
	glEnd();
	glPopMatrix();

	// Disable Texturig
	glDisable(GL_TEXTURE_2D);


}
// Extract Slice from the 3D Spectrum
void GetSpectrumSlice()
{
	float fraction = 0.00390625;
	//////////////////////////////////////////////////////////////////////// Slice 0
	GetSlice(trans/256, &mFBO_ID, &mSliceTextureSrcID);

	// Attach FBO
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, mFBO_ID);
	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);

	// Writing FBO Texture Components into mFrameBufferArray
	glReadPixels(0,0, mSliceWidth, mSliceHeight, RG, GL_FLOAT, mFrameBufferArray);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);


	/*
	//////////////////////////////////////////////////////////////////////// Slice -1
	GetSlice(-fraction, &mFBO_ID, &mSliceTextureSrcID);

	// Attach FBO
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, mFBO_ID);
	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);

	// Writing FBO Texture Components into mFrameBufferArray
	glReadPixels(0,0, mSliceWidth, mSliceHeight, RG, GL_FLOAT, mFrameBufferArray_N1);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

	//////////////////////////////////////////////////////////////////////// Slice -2
	GetSlice((-2 * fraction), &mFBO_ID, &mSliceTextureSrcID);

	// Attach FBO
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, mFBO_ID);
	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);

	// Writing FBO Texture Components into mFrameBufferArray
	glReadPixels(0,0, mSliceWidth, mSliceHeight, RG, GL_FLOAT, mFrameBufferArray_N2);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

	//////////////////////////////////////////////////////////////////////// Slice +1
	GetSlice(fraction, &mFBO_ID, &mSliceTextureSrcID);

	// Attach FBO
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, mFBO_ID);
	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);

	// Writing FBO Texture Components into mFrameBufferArray
	glReadPixels(0,0, mSliceWidth, mSliceHeight, RG, GL_FLOAT, mFrameBufferArray_P1);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

	//////////////////////////////////////////////////////////////////////// Slice +2
	GetSlice(2 * fraction, &mFBO_ID, &mSliceTextureSrcID);

	// Attach FBO
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, mFBO_ID);
	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);

	// Writing FBO Texture Components into mFrameBufferArray
	glReadPixels(0,0, mSliceWidth, mSliceHeight, RG, GL_FLOAT, mFrameBufferArray_P2);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	*/

	/*
	// Check the Extracted Slices
	for (int i = 0; i < 256; i++)
	{
		printf("%f | %f | %f | %f | %f \n", mFrameBufferArray_N2[i], mFrameBufferArray_N1[i], mFrameBufferArray[i], mFrameBufferArray_P1[i], mFrameBufferArray_P2[i]);
	}
	*/

	/*
	// Pack Slices in 3D array
	int mCtr = 0;
	for (int i = 0; i < 256; i++)
		for(int j = 0; j < 256; j++)
		{
			mPlane3D[i][j][1].x = mFrameBufferArray_N2[mCtr];
			mPlane3D[i][j][1].y = mFrameBufferArray_N2[mCtr+1];
			mPlane3D[i][j][2].x = mFrameBufferArray_N1[mCtr];
			mPlane3D[i][j][2].y = mFrameBufferArray_N1[mCtr+1];
			mPlane3D[i][j][3].x = mFrameBufferArray[mCtr];
			mPlane3D[i][j][3].y = mFrameBufferArray[mCtr+1];
			mPlane3D[i][j][4].x = mFrameBufferArray_P1[mCtr];
			mPlane3D[i][j][4].y = mFrameBufferArray_P1[mCtr+1];
			mPlane3D[i][j][5].x = mFrameBufferArray_P2[mCtr];
			mPlane3D[i][j][5].y = mFrameBufferArray_P2[mCtr+1];

			mCtr += 2;
	}
	*/

	/*
	// Do Slice Resampling Along a Plane
	int ctr = 0;
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			for (int k = 0; k < 5; k++)
			{
				mPlane3D_Out[i][j][2].x += mPlane3D[i][j][k].x;
				mPlane3D_Out[i][j][2].y += mPlane3D[i][j][k].y;
			}
		}
	}

	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			mPlane3D[i][j][2].x = mPlane3D_Out[i][j][2].x /= 5;
			mPlane3D[i][j][2].y = mPlane3D_Out[i][j][2].y /= 5;
		}
	}
	*/


	/*
	// Do it for the Plane (2D Lattice)
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			mPlane3D_Out[i][j][2].x = 0;
			mPlane3D_Out[i][j][2].y = 0;
		}
	}


	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{

			mPlane3D_Out[i][j][2].x += mPlane3D[i][j][2].x;
			mPlane3D_Out[i][j][2].y += mPlane3D[i][j][2].y;

			if (!(i-2 < 0))
			{
				mPlane3D_Out[i][j][2].x += mPlane3D[i-2][j][2].x;
				mPlane3D_Out[i][j][2].y += mPlane3D[i-2][j][2].y;
			}

			if (!(i-1 < 0))
			{
				mPlane3D_Out[i][j][2].x += mPlane3D[i-1][j][2].x;
				mPlane3D_Out[i][j][2].y += mPlane3D[i-1][j][2].y;
			}

			if (!(i+1 > 255))
			{
				mPlane3D_Out[i][j][2].x += mPlane3D[i+1][j][2].x;
				mPlane3D_Out[i][j][2].y += mPlane3D[i+1][j][2].y;
			}

			if (!(i+2 > 255))
			{
				mPlane3D_Out[i][j][2].x += mPlane3D[i+2][j][2].x;
				mPlane3D_Out[i][j][2].y += mPlane3D[i+2][j][2].y;
			}

			/////////////////////////////////////////////////////////

			if (!(j-2 < 0))
			{
				mPlane3D_Out[i][j][2].x += mPlane3D[i][j-2][2].x;
				mPlane3D_Out[i][j][2].y += mPlane3D[i][j-2][2].y;
			}

			if (!(j-1 < 0))
			{
				mPlane3D_Out[i][j][2].x += mPlane3D[i][j-1][2].x;
				mPlane3D_Out[i][j][2].y += mPlane3D[i][j-1][2].y;
			}

			if (!(j+1 > 255))
			{
				mPlane3D_Out[i][j][2].x += mPlane3D[i][j+1][2].x;
				mPlane3D_Out[i][j][2].y += mPlane3D[i][j+1][2].y;
			}

			if (!(j+2 > 255))
			{
				mPlane3D_Out[i][j][2].x += mPlane3D[i][j+2][2].x;
				mPlane3D_Out[i][j][2].y += mPlane3D[i][j+2][2].y;
			}

			if ((i-1 < 0) || (i-2 < 0) || (j-1 < 0) || (j-2 < 0) || (i+1 > 255) || (i+2>255) || (j+1>255) || (j+2>255))
			{
				mPlane3D_Out[i][j][2].x = mPlane3D[i][j][2].x;
				mPlane3D_Out[i][j][2].y = mPlane3D[i][j][2].y;
			}

			else
			{
				mPlane3D_Out[i][j][2].x = (mPlane3D[i][j][2].x +  mPlane3D[i-2][j][2].x + mPlane3D[i-1][j][2].x + mPlane3D[i+1][j][2].x + mPlane3D[i+2][j][2].x);
				mPlane3D_Out[i][j][2].y = (mPlane3D[i][j][2].y + mPlane3D[i-2][j][2].y + mPlane3D[i-1][j][2].y + mPlane3D[i+1][j][2].y + mPlane3D[i+2][j][2].y);

				mPlane3D_Out[i][j][2].x = (mPlane3D[i][j][2].x + mPlane3D[i][j-2][2].x + mPlane3D[i][j-1][2].x + mPlane3D[i][j+1][2].x + mPlane3D[i][j+2][2].x);
				mPlane3D_Out[i][j][2].y = (mPlane3D[i][j][2].y + mPlane3D[i][j-2][2].y + mPlane3D[i][j-1][2].y + mPlane3D[i][j+1][2].y + mPlane3D[i][j+2][2].y);
			}
		}
	}



	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			if (!(i-2 < 0))
			{
				mPlane3D_Out[i][j][2].x /= 3;
				mPlane3D_Out[i][j][2].y /= 3;
			}

			else if (!(i-1 < 0))
			{
				mPlane3D_Out[i][j][2].x /= 4;
				mPlane3D_Out[i][j][2].y /= 4;
			}

			else if (!(i+1 > 255))
			{
				mPlane3D_Out[i][j][2].x /= 4;
				mPlane3D_Out[i][j][2].y /= 4;
			}

			else if (!(i+2 > 255))
			{
				mPlane3D_Out[i][j][2].x /= 3;
				mPlane3D_Out[i][j][2].y /= 3;
			}

			else
			{
				mPlane3D_Out[i][j][2].x /= 5;
				mPlane3D_Out[i][j][2].y /= 5;
			}
		}
	}


	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			if (!(j-2 < 0))
			{
				mPlane3D_Out[i][j][2].x /= 3;
				mPlane3D_Out[i][j][2].y /= 3;
			}

			else if (!(j-1 < 0))
			{
				mPlane3D_Out[i][j][2].x /= 4;
				mPlane3D_Out[i][j][2].y /= 4;
			}

			else if (!(j+1 > 255))
			{
				mPlane3D_Out[i][j][2].x /= 4;
				mPlane3D_Out[i][j][2].y /= 4;
			}

			else if (!(j+2 > 255))
			{
				mPlane3D_Out[i][j][2].x /= 3;
				mPlane3D_Out[i][j][2].y /= 3;
			}

			else
			{
				mPlane3D_Out[i][j][2].x /= 5;
				mPlane3D_Out[i][j][2].y /= 5;
			}
		}
	}
*/

	/*
	// Working ony with the Central 2D Lattice
	ctr = 0;
	for (int i = 0; i < 256; i++)
			for (int j = 0; j < 256; j++)
			{
				mFrameBufferArray[ctr++] = mPlane3D[i][j][2].x;
				mFrameBufferArray[ctr++] = mPlane3D[i][j][2].y;
			}
	*/


	int ctr = 0;
	if (CUDA_ENABLED)
	{
		// CUDA Slice Processing
		ProcessSliceCUDA(mSliceWidth,mSliceHeight);

		printf("CUDA Processing \n");

		for (int i = 0; i < (mSliceWidth * mSliceHeight); i++)
		{
			mSliceArrayComplex[ctr][0] = fReconstructedImage[i].x;
			mSliceArrayComplex[ctr][1] = fReconstructedImage[i].y;
			ctr++;
		}
	}
	else
	{
		printf("CPU Processing \n");


		for (int i = 0; i < (256 * 256 * 2); i += 2)
		{
			mSliceArrayComplex[ctr][0] = mFrameBufferArray[i];
			mSliceArrayComplex[ctr][1] = mFrameBufferArray[i+1];
			ctr++;
		}

		// Filter CPU
		ProcessSlice();
	}

	// 2D FFT
	// printf("Executing 2D Inverse FFT - Begin.... \n");
	mFFTWPlan = fftwf_plan_dft_2d(256, 256, mSliceArrayComplex,mSliceArrayComplex , FFTW_BACKWARD, FFTW_ESTIMATE);
	fftwf_execute(mFFTWPlan);
	//* printf("Executing 2D Inverse FFT - End.... \n");

	// Scaling
	int mSliceSize = mVolWidth * mVolHeight;
	int mNormalizedVal = mSliceSize * mScalingFactor * 1;
	for (int sliceCtr = 0; sliceCtr < mSliceSize; sliceCtr++)
	{
		mAbsoluteReconstructedImage[sliceCtr] =  (float) sqrt((mSliceArrayComplex[sliceCtr][0] * mSliceArrayComplex[sliceCtr][0]) + (mSliceArrayComplex[sliceCtr][1] * mSliceArrayComplex[sliceCtr][1]))/(mNormalizedVal);
	}

	ctr = 0;
	for (int i = 0; i < mVolWidth; i++)
		for(int j = 0; j < mVolHeight; j++)
		{
			mImg_2D_Temp[i][j] = mAbsoluteReconstructedImage[ctr];
			mImg_2D[i][j] = 0;
			ctr++;
		}

	//* printf("Wrapping Around Resulting Image - Begin.... \n");
	mImg_2D = FFT_Shift_2D(mImg_2D_Temp, mImg_2D, mUniDim);
	mAbsoluteReconstructedImage = Repack_2D(mImg_2D, mAbsoluteReconstructedImage, mUniDim);
	//* printf("Wrapping Around Resulting Image - End.... \n");

	for (int i = 0; i < mVolWidth * mVolHeight; i++)
	{
		mRecImage[i] = (uchar)(mAbsoluteReconstructedImage[i]);
	}

	// glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

	// Create 2D Texture Object as a Render Target
	glGenTextures(1, &mSliceTextureID);
	glBindTexture(GL_TEXTURE_2D, mSliceTextureID);

	// 2D Texture Creation & Parameters
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	// Automatic Mipmap Generation Included in OpenGL v1.4
	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, mSliceWidth, mSliceHeight, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, mRecImage);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void OpenGL::displayProjection()
{

}
