# This pyhton script can be used to generate the .npy files with the spectra used to calculate the yearly energy yield

import tandems

# Use spectra from National Solar Resource DataBase
tandems.generate_spectral_bins(NSRDBfile='Reno.csv')

# Use spectra from National Solar Resource DataBase
tandems.generate_spectral_bins(NSRDBfile='Boulder.csv')

# Use spectra from National Solar Resource DataBase
tandems.generate_spectral_bins(NSRDBfile='Indianapolis.csv')

# Use spectra from National Solar Resource DataBase
tandems.generate_spectral_bins(NSRDBfile='Philadelphia.csv')

# Fixed lattitude, tracking for both global and direct spectra, random AOD and PW
tandems.generate_spectral_bins(latMin=40,latMax=40,fname='lat40') 

# No tracking for global spectra, plane tilted 37 degrees
tandems.generate_spectral_bins(latMin=40,latMax=40,tracking='38 37 180',fname='lat40tilted')

# 1 axis azimutal tracking for global spectra, plane tilted 37 degrees
tandems.generate_spectral_bins(latMin=40,latMax=40,tracking='38 37 -999',fname='lat401axis')

# Lattitude = 60
tandems.generate_spectral_bins(latMin=60,latMax=60,fname='lat60')

# Lattitude = 0
tandems.generate_spectral_bins(latMin=0,latMax=0,fname='lat0')

# Fixed AOD and PW
tandems.generate_spectral_bins(latMin=40,latMax=40,fname='lat40lowAOD',AOD=0.084,PW=1.416)

# Random locations with -50 < Lattitude < 50
tandems.generate_spectral_bins(latMin=-50,latMax=50,fname='randomLocations')

# FUNCTION tandems.generate_spectral_bins(latMin=40 , latMax=40, longitude='random', AOD='random', PW='random', tracking=True, NSRDBfile='', fname='Iscs')

#    Generates a file with a complete set of binned averaged spectra for yearly energy yield calculations
#    Location can be randomized within given lattitude limits: latMin, latMax.
#    This might be useful to optimize cell designs to yield maximum energy for a range of lattitudes and atmospheric conditions, rather than at a single specific location.

#    NECESARY CHANGES IN SMARTS 2.9.5 SOURCE CODE
    # Line 189
    #       batch = .TRUE.
    # Line 1514
    #      IF(Zenit.LE.75.5224)GOTO 13
    #      WRITE(16,103,iostat = Ierr24)Zenit
    # 103  FORMAT(//,'Zenit  =  ',F6.2,' is > 75.5224 deg. (90 in original code) This is equivalent to AM < 4'
#    This change is needed because trackers are shadowed by neighboring trackers when the sun is near the horizon. 
#    Zenit 80 is already too close to the horizon to use in most cases due to shadowing issues.
