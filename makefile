FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbodyP: nbody.o compute_parallel.o
	nvcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.cu planets.h config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< 
compute_parallel.o: compute_parallel.cu config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $<
clean:
	rm -f *.o nbody
	rm -f *.o nbodyP
