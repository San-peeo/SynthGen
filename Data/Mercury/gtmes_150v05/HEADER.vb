PDS_VERSION_ID             = PDS3
RECORD_TYPE                = FIXED_LENGTH
FILE_RECORDS               = 11478
RECORD_BYTES               = 122
FILE_NAME                  = "GTMES_150V05_SHA.TAB"
^SHADR_HEADER_TABLE        = ("GTMES_150V05_SHA.TAB",1)
^SHADR_COEFFICIENTS_TABLE  = ("GTMES_150V05_SHA.TAB",3)
INSTRUMENT_HOST_NAME       = "MESSENGER"
TARGET_NAME                = "MERCURY"
INSTRUMENT_NAME            = "MERCURY LASER ALTIMETER"
DATA_SET_ID                = "MESS-H-RSS/MLA-5-SDP-V1.0"
OBSERVATION_TYPE           = "TOPOGRAPHY"
PRODUCT_ID                 = "GTMES_150V05_SHA"
SOURCE_PRODUCT_ID          = {"fit1em2_150.out",
                             "pck00010_msgr_v21.tpc",
                             "Release15_shape.csv"}
COORDINATE_SYSTEM_NAME     = PLANETOCENTRIC
START_TIME                 = 2008-01-14T19:03:02
STOP_TIME                  = 2015-04-30T11:13:55
PRODUCT_CREATION_TIME      = 2016-01-10
PRODUCT_RELEASE_DATE       = 2016-05-06
PRODUCER_FULL_NAME         = "DAVID E. SMITH"
PRODUCER_INSTITUTION_NAME  = "NASA/GODDARD SPACE FLIGHT CENTER"
PRODUCT_VERSION_TYPE       = "PRELIMINARY"
PRODUCER_ID                = MESSENGER_MLA_TEAM
SOFTWARE_NAME              = "N/A"

DESCRIPTION                = "
  This file contains coefficients for a spherical harmonic model
  of the shape of Mercury as radii with respect to the center of mass.
  The solution is a weighted least-squares fit to 26 million MLA measurements
  of radius averaged into 0.5x0.5 degree bins, with the inclusion of 557 
  radio occultation measurements of radius, 359 of which are in the southern
  hemisphere. The MLA data are weighted at a standard deviation of 0.3 km,
  divided by the square root of the number of observations in each bin,
  while the occultations are weighted at 0.1 km standard deviation.
  A power law constraint is applied to damp the solution variance.
  The normalization commonly used in geophysics is employed,
  where the integral over the sphere of each harmonic squared is 4 pi.
  The maximum degree and order is 150, but should be truncated to 148 to
  avoid noticeable aliasing in the last two degrees."  
  
OBJECT                     = SHADR_HEADER_TABLE
ROWS                         = 1
COLUMNS                      = 8
ROW_BYTES                    = 137
ROW_SUFFIX_BYTES             = 107
INTERCHANGE_FORMAT           = ASCII
DESCRIPTION                  = "The SHADR header includes descriptive
 information about the spherical harmonic coefficients which follow in
 SHADR_COEFFICIENTS_TABLE.  The header consists of a single record of eight
 (delimited) data columns requiring 137 bytes, a pad of 105 unspecified ASCII
 characters, an ASCII carriage-return, and an ASCII line-feed."

  OBJECT                   = COLUMN
    NAME                         = "REFERENCE RADIUS"
    DATA_TYPE                    = ASCII_REAL
    START_BYTE                   = 1
    BYTES                        = 23
    FORMAT                       = "E23.16"
    UNIT                         = "KILOMETER"
    DESCRIPTION                  = "The assumed reference radius of the
     spherical planet."
  END_OBJECT               = COLUMN

  OBJECT                   = COLUMN
    NAME                         = "CONSTANT"
    DATA_TYPE                    = ASCII_REAL
    START_BYTE                   = 25
    BYTES                        = 23
    FORMAT                       = "E23.16"
    UNIT                         = "N/A"
    DESCRIPTION                  = "For a gravity field model the assumed
     gravitational constant GM in km cubed per seconds squared for the
     planet.  For a topography model, set to 1."
  END_OBJECT               = COLUMN

  OBJECT                   = COLUMN
    NAME                         = "UNCERTAINTY IN CONSTANT"
    DATA_TYPE                    = ASCII_REAL
    START_BYTE                   = 49
    BYTES                        = 23
    FORMAT                       = "E23.16"
    UNIT                         = "N/A"
    DESCRIPTION                  = "For a gravity field model the
     uncertainty in the gravitational constant GM in km cubed per seconds
     squared for the planet (or, set to 0 if not known). For a topography
     model, set to 0."
  END_OBJECT               = COLUMN

  OBJECT                   = COLUMN
    NAME                         = "DEGREE OF FIELD"
    DATA_TYPE                    = ASCII_INTEGER
    START_BYTE                   = 73
    BYTES                        = 5
    FORMAT                       = "I5"
    UNIT                         = "N/A"
    DESCRIPTION                  = "Degree of the model field."
  END_OBJECT               = COLUMN

  OBJECT                   = COLUMN
    NAME                         = "ORDER OF FIELD"
    DATA_TYPE                    = ASCII_INTEGER
    START_BYTE                   = 79
    BYTES                        = 5
    FORMAT                       = "I5"
    UNIT                         = "N/A"
    DESCRIPTION                  = "Order of the model field."
  END_OBJECT               = COLUMN

  OBJECT                   = COLUMN
    NAME                         = "NORMALIZATION STATE"
    DATA_TYPE                    = ASCII_INTEGER
    START_BYTE                   = 85
    BYTES                        = 5
    FORMAT                       = "I5"
    UNIT                         = "N/A"
    DESCRIPTION                  = "The normalization indicator.
     For gravity field:
        0   coefficients are unnormalized
        1   coefficients are normalized
        2   other."
  END_OBJECT               = COLUMN

  OBJECT                   = COLUMN
    NAME                         = "REFERENCE LONGITUDE"
    POSITIVE_LONGITUDE_DIRECTION = "EAST"
    DATA_TYPE                    = ASCII_REAL
    START_BYTE                   = 91
    BYTES                        = 23
    FORMAT                       = "E23.16"
    UNIT                         = "DEGREE"
    DESCRIPTION                  = "The reference longitude for the
     spherical harmonic expansion; normally 0."
  END_OBJECT               = COLUMN

  OBJECT                   = COLUMN
    NAME                         = "REFERENCE LATITUDE"
    DATA_TYPE                    = ASCII_REAL
    START_BYTE                   = 115
    BYTES                        = 23
    FORMAT                       = "E23.16"
    UNIT                         = "DEGREE"
    DESCRIPTION                  = "The reference latitude for the
     spherical harmonic expansion; normally 0."
  END_OBJECT               = COLUMN

END_OBJECT                 = SHADR_HEADER_TABLE

OBJECT                     = SHADR_COEFFICIENTS_TABLE
  ROWS                       = 11476
  COLUMNS                    = 6
  ROW_BYTES                  = 107
  ROW_SUFFIX_BYTES           = 15
  INTERCHANGE_FORMAT         = ASCII
  DESCRIPTION                = "The SHADR coefficients table contains the
   coefficients for the spherical harmonic model. Each row in the table
   contains the degree index m, the order index n, the coefficients Cmn and
   Smn, and the uncertainties in Cmn and Smn. The (delimited) data require
   107 ASCII characters; these are followed by a pad of 13 unspecified ASCII
   characters, an ASCII carriage-return, and an ASCII line-feed."

  OBJECT                   = COLUMN
    NAME                         = "COEFFICIENT DEGREE"
    DATA_TYPE                    = ASCII_INTEGER
    START_BYTE                   = 1
    BYTES                        = 5
    FORMAT                       = "I5"
    UNIT                         = "N/A"
    DESCRIPTION                  = "The degree index m of the C and S
     coefficients in this record."
  END_OBJECT               = COLUMN

  OBJECT                   = COLUMN
    NAME                         = "COEFFICIENT ORDER"
    DATA_TYPE                    = ASCII_INTEGER
    START_BYTE                   = 7
    BYTES                        = 5
    FORMAT                       = "I5"
    UNIT                         = "N/A"
    DESCRIPTION                  = "The order index n of the C and S
     coefficients in this record."
  END_OBJECT               = COLUMN

  OBJECT                   = COLUMN
    NAME                         = "C"
    DATA_TYPE                    = ASCII_REAL
    START_BYTE                   = 13
    BYTES                        = 23
    FORMAT                       = "E23.16"
    UNIT                         = "N/A"
    DESCRIPTION                  = "The coefficient Cmn for this spherical
     harmonic model."
  END_OBJECT               = COLUMN

  OBJECT                   = COLUMN
    NAME                         = "S"
    DATA_TYPE                    = ASCII_REAL
    START_BYTE                   = 37
    BYTES                        = 23
    FORMAT                       = "E23.16"
    UNIT                         = "N/A"
    DESCRIPTION                  = "The coefficient Smn for this spherical
     harmonic model."
  END_OBJECT               = COLUMN

  OBJECT                   = COLUMN
    NAME                         = "C UNCERTAINTY"
    DATA_TYPE                    = ASCII_REAL
    START_BYTE                   = 61
    BYTES                        = 23
    FORMAT                       = "E23.16"
    UNIT                         = "N/A"
    DESCRIPTION                  = "The uncertainty in the coefficient Cmn
     for this spherical harmonic model."
  END_OBJECT               = COLUMN

  OBJECT                   = COLUMN
    NAME                         = "S UNCERTAINTY"
    DATA_TYPE                    = ASCII_REAL
    START_BYTE                   = 85
    BYTES                        = 23
    FORMAT                       = "E23.16"
    UNIT                         = "N/A"
    DESCRIPTION                  = "The uncertainty in the coefficient Smn
     for this spherical harmonic model."
  END_OBJECT               = COLUMN

END_OBJECT           = SHADR_COEFFICIENTS_TABLE

END