#!/usr/bin/python

import subprocess

executable = ""
executable += "./Build/iB_ComplexDoubleRnd_FFTShift_2D"
executable += "./Build/iB_ComplexDoubleRnd_FFTShift_3D"
executable += "./Build/iB_ComplexDoubleSeq_FFTShift_2D"
executable += "./Build/iB_ComplexDoubleSeq_FFTShift_3D"
executable += "./Build/iB_ComplexSingleRnd_FFTShift_2D"
executable += "./Build/iB_ComplexSingleRnd_FFTShift_3D"
executable += "./Build/iB_ComplexSingleSeq_FFTShift_2D"
executable += "./Build/iB_ComplexSingleSeq_FFTShift_3D"
executable += "./Build/iB_RealDoubleRnd_FFTShift_2D"
executable += "./Build/iB_RealDoubleRnd_FFTShift_3D"
executable += "./Build/iB_RealDoubleSeq_FFTShift_2D"
executable += "./Build/iB_RealDoubleSeq_FFTShift_3D"
executable += "./Build/iB_RealSingleRnd_FFTShift_2D"
executable += "./Build/iB_RealSingleRnd_FFTShift_3D"
executable += "./Build/iB_RealSingleSeq_FFTShift_2D"
executable += "./Build/iB_RealSingleSeq_FFTShift_3D"

print(executable)
subprocess.call(executable, shell=True) 




