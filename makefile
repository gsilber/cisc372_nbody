FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

# parallel
parallel: nbody_p.o compute_p.o
	nvcc $(FLAGS) -fmad=false $^ -o $@ $(LIBS)
nbody_p.o: nbody.cpp planets.h config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -fmad=false -c $< -o nbody_p.o
compute_p.o: compute.cu config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -fmad=false -c $< -o compute_p.o

# serial
serial: nbody_s.o compute_s.o
	gcc $(FLAGS) $^ -o $@ $(LIBS)
nbody_s.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $< -o nbody_s.o
compute_s.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $< -o compute_s.o

# clean
clean:
	rm -f *.o nbody parallel serial
