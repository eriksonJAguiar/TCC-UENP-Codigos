SRC= model.py imodel.py 
REST= README makefile sequence.dat


#interactive
demo: 	Py_imodel.py
	python Py_imodel.py

fit:	X_demo.m
	octave X_demo


clean:
	\rm -r -f *.pyc 
cleanall: 
	rm -rf  *.pyc ./_out/


