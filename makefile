FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbody: nbody.o compute.o cuda.o
	nvcc $(FLAGS) $^ -o $@ $(LIBS)

nbody.o: nbody.c planets.h config.h vector.h cuda.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $< 

compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $< 

cuda.o: cuda.cu cuda.h
	nvcc $(FLAGS) -c $<

clean:
	rm -f *.o nbody