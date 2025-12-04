Don't introduce AI generated slop such as:

- Extra comments that a human wouldn't add or is inconsistent with the rest of the file
- Extra defensive checks or try/catch blocks that are abnormal for that area of the codebase
- Casts to any to get around type issues
- Variables that are only used a single time after declaration, prefer inlining the rhs
- Any other style that is inconsistent with the file
- Check whether we shalln't first update existing files for a new functionality rather than creating new files each time

Report at the end with only a 1-3 sentence summary what you changed, don't create huge markdowns that extrapolate potential results. 