# Kayaking vs. YoYo split

A two-class subset of UCF-101 for the binary experiment, with the same columns as the official
lists in [`../ucf101`](../ucf101):

- `classInd.txt`: the two classes, Kayaking (1) and YoYo (2).
- `trainlist01.txt`: 250 videos from groups 1 to 23 of both classes, each with its class number
  (131 Kayaking, 119 YoYo).
- `testlist01.txt`: 19 videos from groups 24 and 25 of both classes (10 Kayaking, 9 YoYo).

Together the two lists cover all 269 Kayaking and YoYo videos. UCF-101 sorts the videos of each
class into 25 groups, and videos in the same group can share the same background, viewpoint and
people, so each group sits entirely on one side of the split. The 2022 version of this split had
one video of YoYo group 23 (`v_YoYo_g23_c06.avi`) in the test list while the other five videos
of that group were in the training list; it is now in the training list.

The split is not official split 1, which tests on groups 1 to 7.
