# Fork creator:   Codename;0
**You can contact on discord:    .codename0.**

## Codename's fork features:
| Training related |

- Mangio's crepe F0/Feature extraction method
It is the best ( Not the fastest tho. ) extraction method for models that rely on clean ( No reverb, delay, harmonies, noise etc. ) or truly HQ datasets.
- Adjustable hop_length for Mangio's crepe method ( Definitely the biggest perk of using Mangio's crepe. )
- Changed the default hop_length for rmvpe to 64ms.
- Envelopes for processed samples / segments to avoid zero-crossing ( waveform interruption ) clicks.
- My own " Mel similarity metric " as a bonus. Helps in spotting overtraining / overfitting and mode collapses. Metric is being displayed in the console / log and is also logged in tensorboard files.
- Sox is used for resampling ( vhq - Very High quality as default )

.

| Inference related |

- Automatic index file matching for models ( tho I can't seem to get it work on my end. It's perhaps either buggy or requires certain conditions ) 
- Auto detection of audio files in " audios " folder + a dropdown to choose 'em. ( Much easier than having to copy/paste paths to your .wav or whatever all the time )
- Changed default Feature index search ratio to '0.5'.
- Mangio-crepe ( standard; " full " ) - Go to method.


.

| ONNX Export related |
- There's been many issues and problems with newest releases' onnx exporting so I decided to port 0618's ( RVC 0618 v2 beta release ) way of handling it.
( tl;dr, now exporting works. It exports the onnx models with onnx 14's operators.  ~ Tested having onnx 15 runtime - so the latest one available. )

I've confirmed the exported .onnx models to work with;
- my "RVC_Onnx_Infer" thingy: https://github.com/codename0og/RVC_Onnx_Infer ( and generally rvc's own original onnx inference demo ) 
- As for w-okada, nope. Shapes mismatch.
( I suppose that's pretty normal and in order to use w-okada with onnx, you gotta convert the rvc's .pth to .onnx in the w-okada itself. )


.

## Bugs / Stuff that's broken.

- Noticed that after you unload the model from inference the " protect voiceless consonants " value becomes null.
FIX: After you unload the voice / model and then load any again, simply move the slider or input the default value: 0.33
( Now I am not sure whether it always was like that in mangio or not, been ages. )

- No other bugs or issues I've captured. In case you do and it's serious, write me a msg on discord.


## Potentially to be added in future - WIP and unsure if I wanna go that way.
- Native ONNX inference
I kinda worked on it but mehh, due to the reasons I've written about in my onnx infer thingy I sorta don't feel a need to work on it, at least for now.
