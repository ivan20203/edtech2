
import os
import sys
import datetime
import time

from APIs import TTM, TTS, TTA, MIX, CAT, COMPUTE_LEN


fg_audio_lens = []
wav_path = "/home/ivan/github-repos/edtech/edtech/PodAgent/output/sessions/test/audio"
os.makedirs(wav_path, exist_ok=True)


TTM(text="Upbeat, inspiring introductory talk show theme music", length=30, volume=-35, out_wav=os.path.join(wav_path, "fg_music_0_Upbeat_inspiring_introductory_talk_show.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_music_0_Upbeat_inspiring_introductory_talk_show.wav")))

TTS(text="Good evening, everyone! Welcome to another thought-provoking episode of our talk show. I'm your host, and today we have some esteemed guests with us to delve into the intriguing world of plasma physics.", speaker_id="0", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_0_Good_evening_everyone_Welcome_to.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7507_100463_000003_000002.wav", speaking_style="Enthusiastic and welcoming")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_0_Good_evening_everyone_Welcome_to.wav")))

TTA(text="Audience clapping", length=3, volume=-35, out_wav=os.path.join(wav_path, "fg_sound_effect_0_Audience_clapping.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_0_Audience_clapping.wav")))

TTS(text="Please welcome Dr. Albert Kohlman, a Plasma Physicist and a leading innovator in the field. Dr. Kohlman has been instrumental in furthering our understanding of plasma behavior, particularly in Tokamak, and has been honored with the E. O. Lawrence Award. Dr. Kohlman, we're thrilled to have you with us.", speaker_id="0", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_1_Please_welcome_Dr_Albert_Kohlman.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7507_100463_000003_000002.wav", speaking_style="Friendly, respectful")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_1_Please_welcome_Dr_Albert_Kohlman.wav")))

TTS(text="Thank you for having me. Plasmas, their stability and complex behavior have become a passion, allowing me to contribute significantly to the field, particularly regarding the influence of electron and ion diamagnetic drifts on kink modes in plasmas.", speaker_id="195", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_2_Thank_you_for_having_me.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/6882_79975_000034_000000.wav", speaking_style="Passionate, scholarly")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_2_Thank_you_for_having_me.wav")))

TTA(text="Glass clinks lightly as guest settle in", length=2, volume=-35, out_wav=os.path.join(wav_path, "fg_sound_effect_1_Glass_clinks_lightly_as_guest.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_1_Glass_clinks_lightly_as_guest.wav")))

TTS(text="Intriguing! You've talked about the myriad challenges, including the complex dynamics brought by these drifts and the predictive accuracy for larger systems like ITER. Could you elaborate more on this?", speaker_id="0", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_3_Intriguing_Youve_talked_about_the.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7507_100463_000003_000002.wav", speaking_style="Engaged, curious")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_3_Intriguing_Youve_talked_about_the.wav")))

TTS(text="Absolutely. Plasma stability in high-temperature conditions is a central challenge. Understanding diamagnetic drifts' influence is vital for improving plasma confinement and the efficiency of fusion power reactors. Moreover, correlating theoretical predictions with experimental observations is critical for our work.", speaker_id="195", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_4_Absolutely_Plasma_stability_in_hightemperature.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/6882_79975_000034_000000.wav", speaking_style="Detailed, informative")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_4_Absolutely_Plasma_stability_in_hightemperature.wav")))

TTA(text="Audience clapping", length=2, volume=-35, out_wav=os.path.join(wav_path, "fg_sound_effect_2_Audience_clapping.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_2_Audience_clapping.wav")))

TTS(text="That's fascinating, Dr. Kohlman. Indeed, understanding these complexities opens avenues for advancements in plasma physics. But now, let's introduce our second guest. Please welcome Dr. Emily Wang, an accomplished Astrophysicist with significant contributions to plasma behavior in space. Welcome, Dr. Wang.", speaker_id="0", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_5_Thats_fascinating_Dr_Kohlman_Indeed.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7507_100463_000003_000002.wav", speaking_style="Respectful, enthusiastic")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_5_Thats_fascinating_Dr_Kohlman_Indeed.wav")))

TTS(text="Hello! It's delightful to be here. I've devoted much of my studies to understand the relationship between diamagnetic drifts and kink mode instabilities, contributing to plasma stability and successful magnetic confinement fusion.", speaker_id="144", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_6_Hello_Its_delightful_to_be.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/1263_138246_000020_000000.wav", speaking_style="Delighted, knowledgeable")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_6_Hello_Its_delightful_to_be.wav")))

TTS(text="Indeed, your research is groundbreaking. In light of the challenges discussed, could you share your thoughts on the impact of micro-turbulence on plasma confinement and usage of advanced materials in Tokamak?", speaker_id="0", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_7_Indeed_your_research_is_groundbreaking.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7507_100463_000003_000002.wav", speaking_style="Inquisitive, interested")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_7_Indeed_your_research_is_groundbreaking.wav")))

TTS(text="Certainly. Micro-turbulence profoundly impacts plasma behavior, understanding which can improve plasma control significantly, while advanced materials too introduce new dynamics into plasma interaction with reactor walls. Addressing these challenges holds the key to better plasma stability and increased fusion power output.", speaker_id="144", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_8_Certainly_Microturbulence_profoundly_impacts_plasma.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/1263_138246_000020_000000.wav", speaking_style="Authoritative, explanatory")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_8_Certainly_Microturbulence_profoundly_impacts_plasma.wav")))

TTA(text="Contemplative 'hmm' from the audience", length=2, volume=-35, out_wav=os.path.join(wav_path, "fg_sound_effect_3_Contemplative_hmm_from_the_audience.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_3_Contemplative_hmm_from_the_audience.wav")))

TTS(text="The potential in this field is immense. Both of you suggested future directions for this field, including elaborate modeling techniques and better understanding of diamagnetic drifts. Do you believe that this research can yield significant breakthroughs in the future?", speaker_id="0", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_9_The_potential_in_this_field.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7507_100463_000003_000002.wav", speaking_style="Probing, hopeful")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_9_The_potential_in_this_field.wav")))

TTA(text="Quick intake of breath conveying interest", length=1, volume=-35, out_wav=os.path.join(wav_path, "fg_sound_effect_4_Quick_intake_of_breath_conveying.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_4_Quick_intake_of_breath_conveying.wav")))

TTS(text="Absolutely. Improving our understanding and predicting plasma behavior can make fusion power more feasible as an energy source for future generations.", speaker_id="195", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_10_Absolutely_Improving_our_understanding_and.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/6882_79975_000034_000000.wav", speaking_style="Reassuring, confident")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_10_Absolutely_Improving_our_understanding_and.wav")))

TTA(text="Nodding agreement 'Mmm-hmms' from audience", length=2, volume=-35, out_wav=os.path.join(wav_path, "fg_sound_effect_5_Nodding_agreement_Mmmhmms_from_audience.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_5_Nodding_agreement_Mmmhmms_from_audience.wav")))

TTS(text="I concur. Be it exploring different drift scenarios or controlling plasma turbulence, each advancement brings us closer to sustaining fusion power, particularly in advanced Tokamak designs like ITER.", speaker_id="144", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_11_I_concur_Be_it_exploring.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/1263_138246_000020_000000.wav", speaking_style="Assertive, encouraging")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_11_I_concur_Be_it_exploring.wav")))

TTS(text="Well, that concludes our enlightening discussion for today. I would like to extend my deepest gratitude to Dr. Kohlman and Dr. Wang for their invaluable insights. Join us next time as we continue to explore the frontiers of science. Goodnight, everyone!", speaker_id="0", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_12_Well_that_concludes_our_enlightening.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7507_100463_000003_000002.wav", speaking_style="Cheerful, grateful")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_12_Well_that_concludes_our_enlightening.wav")))

TTM(text="Upbeat, cheerful closing talk show theme music", length=30, volume=-35, out_wav=os.path.join(wav_path, "fg_music_1_Upbeat_cheerful_closing_talk_show.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_music_1_Upbeat_cheerful_closing_talk_show.wav")))

fg_audio_wavs = []
fg_audio_wavs.append(os.path.join(wav_path, "fg_music_0_Upbeat_inspiring_introductory_talk_show.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_0_Good_evening_everyone_Welcome_to.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_0_Audience_clapping.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_1_Please_welcome_Dr_Albert_Kohlman.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_2_Thank_you_for_having_me.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_1_Glass_clinks_lightly_as_guest.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_3_Intriguing_Youve_talked_about_the.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_4_Absolutely_Plasma_stability_in_hightemperature.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_2_Audience_clapping.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_5_Thats_fascinating_Dr_Kohlman_Indeed.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_6_Hello_Its_delightful_to_be.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_7_Indeed_your_research_is_groundbreaking.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_8_Certainly_Microturbulence_profoundly_impacts_plasma.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_3_Contemplative_hmm_from_the_audience.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_9_The_potential_in_this_field.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_4_Quick_intake_of_breath_conveying.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_10_Absolutely_Improving_our_understanding_and.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_5_Nodding_agreement_Mmmhmms_from_audience.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_11_I_concur_Be_it_exploring.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_12_Well_that_concludes_our_enlightening.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_music_1_Upbeat_cheerful_closing_talk_show.wav"))
CAT(wavs=fg_audio_wavs, out_wav=os.path.join(wav_path, "foreground.wav"))

bg_audio_offsets = []
bg_audio_len = sum(fg_audio_lens[4:20])
bg_audio_offset = sum(fg_audio_lens[:4])
TTA(text="Soft generic talk show background music", volume=-35, length=bg_audio_len, out_wav=os.path.join(wav_path, "bg_sound_effect_0_Soft_generic_talk_show_background.wav"))
bg_audio_offsets.append(bg_audio_offset)

bg_audio_len = sum(fg_audio_lens[10:11])
bg_audio_offset = sum(fg_audio_lens[:10])
TTA(text="Subtle background noise of rustling paper and pencil scribbling", volume=-35, length=bg_audio_len, out_wav=os.path.join(wav_path, "bg_sound_effect_1_Subtle_background_noise_of_rustling.wav"))
bg_audio_offsets.append(bg_audio_offset)

bg_audio_wavs = []
bg_audio_wavs.append(os.path.join(wav_path, "bg_sound_effect_0_Soft_generic_talk_show_background.wav"))
bg_audio_wavs.append(os.path.join(wav_path, "bg_sound_effect_1_Subtle_background_noise_of_rustling.wav"))
bg_audio_wav_offset_pairs = list(zip(bg_audio_wavs, bg_audio_offsets))
bg_audio_wav_offset_pairs.append((os.path.join(wav_path, "foreground.wav"), 0))
MIX(wavs=bg_audio_wav_offset_pairs, out_wav=os.path.join(wav_path, "res_test.wav"))
