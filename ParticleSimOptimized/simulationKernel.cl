#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

//
// particle data structure
//
typedef struct 
{
  double x;
  double y;
  double vx;
  double vy;
  double ax;
  double ay;
} particle_t;

void apply_force_gpu(__global particle_t *particle, __global particle_t *neighbor)
{
  double dx = neighbor->x - particle->x;
  double dy = neighbor->y - particle->y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle->ax += coef * dx;
  particle->ay += coef * dy;

}


// sets all bins sizes to 0.
__kernel void bin_init_gpu(volatile __global long * binSizes, int numberOfBins) {
	int tid = get_global_id(0);
    if(tid >= numberOfBins*numberOfBins) return;
    binSizes[tid] = 0;
	
}

// Bins particles
__kernel void bin_gpu(__global particle_t * particles, int n, __global particle_t * bins, volatile __global long* binSizes, int numberOfBins) {
	int tid = get_global_id(0);
	if(tid >= n) return;

	// Get bin nr for current particle
	int r = particles[tid].x / (2 * cutoff) + 1;
	int t = particles[tid].y / (2 * cutoff) + 1;
	
	int binNr = r * numberOfBins + t;
	// 10 is hardcoded (max bin size)
	int binLoc = r * numberOfBins * 10 + t * 10; 
	
	// atomically add .
	long h = atom_add(&binSizes[binNr], 1);
	bins[binLoc + h] = particles[tid];
}

		
// How do we apply forces is the question.
__kernel void compute_forces_gpu(__global particle_t * particles, 
	int n, __global particle_t * bins, volatile __global long* binSizes, int numberOfBins)
{
  // Get thread (particle) ID
	int tid = get_global_id(0);
	if(tid >= n) return;

	// Get bin nr for current particle
	int r = particles[tid].x / (2 * cutoff) + 1;
	int t = particles[tid].y / (2 * cutoff) + 1;
	
	int binNr = r * numberOfBins + t;
	// 10 is hardcoded (max bin size)
	int binLoc = r * numberOfBins * 10 + t * 10; 

	particles[tid].ax = particles[tid].ay = 0;
	
	// Every bin has 9 neigbors. This is going to be ugly...
	// Left top
	for(int j = 0 ; j < binSizes[binNr-numberOfBins-1] ; j++) {
		apply_force_gpu(&particles[tid], &bins[binLoc - numberOfBins * 10 - 10 + j]);
	}
	
	for(int j = 0 ; j < binSizes[binNr-numberOfBins] ; j++)
		apply_force_gpu(&particles[tid], &bins[binLoc - numberOfBins * 10 + j]);
	
	// Right top
	for(int j = 0 ; j < binSizes[binNr-numberOfBins+1] ; j++)
		apply_force_gpu(&particles[tid], &bins[binLoc - numberOfBins * 10 + 10 + j]);
	
	// Left mid
	for(int j = 0 ; j < binSizes[binNr-1] ; j++)
		apply_force_gpu(&particles[tid], &bins[binLoc - 10 + j]);
	
	// Current bin
	for(int j = 0 ; j < binSizes[binNr] ; j++)
		apply_force_gpu(&particles[tid], &bins[binLoc + j]);
	
	// Right mid
	for(int j = 0 ; j < binSizes[binNr+1] ; j++)
		apply_force_gpu(&particles[tid], &bins[binLoc + 10 + j]);
	
	// Left bot
	for(int j = 0 ; j < binSizes[binNr+numberOfBins-1] ; j++)
		apply_force_gpu(&particles[tid], &bins[binLoc + numberOfBins * 10 - 10 + j]);
	
	// Mid bot
	for(int j = 0 ; j < binSizes[binNr+numberOfBins] ; j++)
		apply_force_gpu(&particles[tid], &bins[binLoc + numberOfBins * 10 + j]);
	
	// Right bot
	for(int j = 0 ; j < binSizes[binNr+numberOfBins+1] ; j++)
		apply_force_gpu(&particles[tid], &bins[binLoc + numberOfBins * 10 + 10 + j]);

}


__kernel void move_gpu (__global particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
  	int tid = get_global_id(0);

  if(tid >= n) return;

    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    particles[tid].vx += particles[tid].ax * dt;
    particles[tid].vy += particles[tid].ay * dt;
    particles[tid].x  += particles[tid].vx * dt;
    particles[tid].y  += particles[tid].vy * dt;

    //
    //  bounce from walls
    //
    while( particles[tid].x < 0 || particles[tid].x > size )
    {
        particles[tid].x  = particles[tid].x < 0 ? -(particles[tid].x) : 2*size-particles[tid].x;
        particles[tid].vx = -(particles[tid].vx);
    }
    while( particles[tid].y < 0 || particles[tid].y > size )
    {
        particles[tid].y  = particles[tid].y < 0 ? -(particles[tid].y) : 2*size-particles[tid].y;
        particles[tid].vy = -(particles[tid].vy);
    }

}
