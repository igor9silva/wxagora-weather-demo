# Codex task: prepare a weather video for today

Run the experimental workflow in `FULL-BODY.md` from this repository. Fetch the latest IPMA day-0 forecast, validate its Lisbon date, generate the four Portuguese narration clips locally, and use the `image_prompt` for each city in the resulting plan to call Codex's built-in image-generation tool once per city. Save each image with the plan's `image_file` name in `assets/generated/<forecast-date>/`. Inspect the images, then register the set with `register_image_set.py`.

Use the existing fictional presenter reference and the local LTX audio-to-video model to render full-body movement for the narration and compose the four scenes with `render_fullbody.py`. Keep graphics compact and leave photo credits and large footers off the video. Verify that each image and temperature matches its forecast, inspect the resulting movement, and provide the video and any important quality limitations. Do not publish to social media. If the required built-in image tool, models, forecast or GPU are unavailable, report that instead of silently reusing stale images or presenting a still as an animated reporter.

This is an agent workflow, not an installed or tested daily schedule.
