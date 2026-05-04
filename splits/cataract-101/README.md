Cataract-101 – Video Dataset of 101 Cataract Surgeries
======================================================

This dataset contains 101 videos of cataract surgeries annotated with two
kinds of information:

* (Anonymous) ID and experience level of operating surgeon
* Starting points of quasi-standardized operation phases in videos

The cataract surgeries have been performed by four different surgeons of
two levels of experience (low, high).

The dataset has been collected and annotated by ophthalmic surgeons of
Klinikum Klagenfurt, and has been prepared and provided by the Institute of
Information Technology (ITEC) of Alpen-Adria-Universitaet (AAU) Klagenfurt, Austria.

The dataset has been published for open access according to the terms of the following
publication, which should be cited upon use of the dataset in further publications:

K. Schoeffmann, M. Taschwer, S. Sarny, B. Münzer, M. J. Primus, D. Putzgruber:
"Cataract-101 – Video Dataset of 101 Cataract Surgeries", ACM Multimedia Systems
Conference (MMSys'18), 2018, to appear. DOI [https://doi.org/10.1145/3204949.3208137](10.1145/3204949.3208137).

## Video Files

Video files are contained in the *videos* subdirectory with file names following
the pattern:

**case_**_videoID_**.mp4**

For example, the file name of Video 269 is *case_269.mp4*.

Video files have been encoded using the H.264/AVC codec with 25 fps and a resolution 
of 720x540 pixels. The mean duration of provided videos is about 12,500 frames 
(8.3 minutes).

## Annotation Files

Annotations are provided in three semicolon-separated values files, each containing
a header line describing the fields.

### videos.csv

Contains metadata pertaining to entire video files, with all integer fields:

* Video ID
* Number of video frames
* Frames per second (25)
* Surgeon ID (1-4)
* Surgeon's experience level (1 = low, 2 = high)

The distribution of videos per surgeon and of videos per level of experience are
shown below.

| Surgeon ID | #Videos |
| ---------- | -------:|
| 1          |      25 |
| 2          |      24 |
| 3          |      32 |
| 4          |      20 |

| Experience | #Videos |
| ---------- | -------:|
| 1 (low)    |      45 |
| 2 (high)   |      56 |

### phases.csv

Describes the ten quasi-standardized operation phases of cataract surgeries
(see the paper cited above for more information). The two fields are:

* Phase ID (integer in the range 1-10)
* Phase name (string)

### annotations.csv

Contains expert annotations of operation phase boundaries of all provided videos.
Due to annotation tool usage, only the starting points of operation phases were
annotated and the accuracy of manually selecting a point in time is not better than
+/- 1 second (+/- 25 frames). Nevertheless, annotations are provided at frame
resolution to facilitate comparison of frame-based evaluations using this dataset.

Each line of the CSV file (except for the header line) describes the starting point
of an operation phase annotation, with the following all integer fields:

* Video ID (matching an ID in *videos.csv*)
* Frame number (zero-based) representing the starting point of an operation phase, 
  which extends to the next annotation or to the end of the same video
* Phase ID (matching an ID in *phases.csv*)

Lines in the CSV file are sorted lexicographically by (video ID, frame number).

Please note the following consequences of the chosen annotation process:

1. The video segment up to the first annotation (start of the first operation
   phase) of a given video is effectively not annotated, i.e. not assigned to
   any operation phase.

2. The video segment starting at the last annotation of a given video effectively
   extends until the end of the video, although the annotated operation phase
   may actually end earlier. However, the video acquisition procedure suggests
   that this "out-of-phase" period at the end of surgeries typically takes only
   a few seconds.

3. Annotations of the same operation phase may occur consecutively within the
   same video, because the same phase is either repeated or divided into sub-phases
   that were also annotated by medical experts. Sub-phases have
   been consolidated post-hoc to conform to the quasi-standardized phases 
   described in *phases.csv*.

Finally, note that the linear sequence of operation phases defined in *phases.csv*
is often not strictly followed in surgeries, mainly for two reasons:

* Phase 2 (viscous agent injection) regularly occurs twice in each cataract surgery,
  and the two video segments are annotated with the same phase ID, because they are 
  usually not distinguishable from a visual perspective.

* Surgeons (especially less experienced ones) may sometimes have to repeat certain 
  phases or sequences of phases.

