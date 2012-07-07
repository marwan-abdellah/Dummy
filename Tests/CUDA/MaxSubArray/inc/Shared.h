#ifndef SHARED_H_
#define SHARED_H_

/* This structure represents the selected value and the coordinates of
 * the max sub array */
struct Max
{
	/* @ Value */
	int val;

	/* Indexes of the sub-array*/
	int x1;
	int y1;
	int x2;
	int y2;
};

#endif
