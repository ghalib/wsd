# Ghalib Suleiman
# Makefile for SENSEVAL-3 scorer

all:
	(cd scorer; make; cd -; cp scorer/scorer sense_scorer;)

clean:
	(rm sense_scorer; cd scorer; make clean; cd -;)
