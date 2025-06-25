
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

TTS(text="Hello and welcome to the Talk Show of Ideas! I'm your host, Steve, and today we have a fascinating topic: spherical geometry. It's more than just a mathematical curiosity; it influences our technology and science in profound ways. To help us explore this, I'm thrilled to have with us two extraordinary guests—Dr. Hannah Cheng, a professor of mathematics with a focus on spherical geometry, and Mr. Alan Martinez, a senior software engineer specializing in geometric algorithms. Welcome both of you to the show!", speaker_id="11", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_0_Hello_and_welcome_to_the.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/5280_204996_000056_000000.wav", speaking_style="enthusiastic and engaging")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_0_Hello_and_welcome_to_the.wav")))

TTA(text="Audience clapping", length=3, volume=-35, out_wav=os.path.join(wav_path, "fg_sound_effect_0_Audience_clapping.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_0_Audience_clapping.wav")))

TTS(text="Thank you, Steve. It's great to be here to talk about spherical geometry, one of the more intriguing branches of mathematics where traditional Euclidean rules don’t quite apply!", speaker_id="82", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_1_Thank_you_Steve_Its_great.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7932_278228_000010_000002.wav", speaking_style="warm and informative")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_1_Thank_you_Steve_Its_great.wav")))

TTS(text="Thanks, Steve! I'm excited to discuss the practical applications of spherical geometry, especially in the tech industry.", speaker_id="114", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_2_Thanks_Steve_Im_excited_to.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7096_80685_000028_000001.wav", speaking_style="friendly and eager")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_2_Thanks_Steve_Im_excited_to.wav")))

TTA(text="Glass clinks lightly as guests settle in", length=2, volume=-35, out_wav=os.path.join(wav_path, "fg_sound_effect_1_Glass_clinks_lightly_as_guests.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_1_Glass_clinks_lightly_as_guests.wav")))

TTS(text="Let's dive right into it. Dr. Cheng, could you start by explaining what distinguishes spherical geometry from regular Euclidean geometry?", speaker_id="11", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_3_Lets_dive_right_into_it.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/5280_204996_000056_000000.wav", speaking_style="curious and open-ended")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_3_Lets_dive_right_into_it.wav")))

TTS(text="Certainly! Spherical geometry deals with figures on the surface of a sphere, where the rules are quite different from those on a flat plane. For example, in Euclidean geometry, parallel lines never meet, but on a sphere, all 'lines'—which we call great circles—eventually intersect. An interesting result of this is that the sum of the angles in a spherical triangle exceeds 180 degrees due to the sphere's curvature.", speaker_id="82", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_4_Certainly_Spherical_geometry_deals_with.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7932_278228_000010_000002.wav", speaking_style="clear and articulate")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_4_Certainly_Spherical_geometry_deals_with.wav")))

TTA(text="Audience clapping", length=2, volume=-35, out_wav=os.path.join(wav_path, "fg_sound_effect_2_Audience_clapping.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_2_Audience_clapping.wav")))

TTS(text="That’s quite fascinating. Mr. Martinez, how do these unique properties of spherical geometry translate into the tech world, particularly in your field?", speaker_id="11", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_5_Thats_quite_fascinating_Mr_Martinez.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/5280_204996_000056_000000.wav", speaking_style="inquisitive and inviting")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_5_Thats_quite_fascinating_Mr_Martinez.wav")))

TTS(text="Great question, Steve! In computer graphics and virtual reality, we often need to accurately render curved surfaces or simulate real-world environments, which inherently involves spherical geometry. For example, when creating 3D models of planets or simulating immersive environments in VR, understanding how to map and calculate on a sphere is critical for realism and precision.", speaker_id="114", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_6_Great_question_Steve_In_computer.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7096_80685_000028_000001.wav", speaking_style="enthusiastic and technical")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_6_Great_question_Steve_In_computer.wav")))

TTS(text="Dr. Cheng, you've mentioned that spherical geometry has a long historical journey. Can you touch on some of the key contributors and how this field has evolved over time?", speaker_id="11", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_7_Dr_Cheng_youve_mentioned_that.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/5280_204996_000056_000000.wav", speaking_style="curious and encouraging")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_7_Dr_Cheng_youve_mentioned_that.wav")))

TTS(text="Of course! The study of spherical geometry dates back to ancient Greece with figures like Euclid and Ptolemy, but significant progress was made during the Islamic Golden Age by scholars such as Al-Khwarizmi. In the modern era, Carl Friedrich Gauss and Bernhard Riemann laid essential groundwork for understanding the intrinsic properties of curved surfaces. Their contributions have influenced both theoretical and practical applications across various scientific fields.", speaker_id="82", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_8_Of_course_The_study_of.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7932_278228_000010_000002.wav", speaking_style="insightful and knowledgeable")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_8_Of_course_The_study_of.wav")))

TTA(text="Audience murmuring in appreciation", length=2, volume=-35, out_wav=os.path.join(wav_path, "fg_sound_effect_3_Audience_murmuring_in_appreciation.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_3_Audience_murmuring_in_appreciation.wav")))

TTS(text="It’s incredible to see how ancient knowledge still impacts modern science. Alan, what’s on the horizon for spherical geometry in terms of future applications and challenges?", speaker_id="11", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_9_Its_incredible_to_see_how.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/5280_204996_000056_000000.wav", speaking_style="forward-looking and engaging")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_9_Its_incredible_to_see_how.wav")))

TTS(text="As technology evolves, we’re seeing more use of spherical geometry in machine learning, where it helps model complex data and improve simulations in geospatial analysis and robotics. However, a major challenge remains in developing efficient algorithms to handle these calculations, particularly for real-time systems, which is crucial for things like augmented reality and more intuitive user interfaces.", speaker_id="114", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_10_As_technology_evolves_were_seeing.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7096_80685_000028_000001.wav", speaking_style="forward-thinking and analytical")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_10_As_technology_evolves_were_seeing.wav")))

TTS(text="To add to that, I believe the intersection of spherical geometry with other non-Euclidean geometries offers exciting potential. It could lead to new theoretical insights and practical applications, especially as we advance in fields like artificial intelligence and data visualization.", speaker_id="82", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_11_To_add_to_that_I.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7932_278228_000010_000002.wav", speaking_style="optimistic and forward-thinking")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_11_To_add_to_that_I.wav")))

TTS(text="Thank you both for such enlightening perspectives. This has been a truly insightful discussion on how spherical geometry shapes not only our understanding of mathematics but also its impact on technology and future innovations. Thank you for joining us today, Dr. Cheng and Mr. Martinez!", speaker_id="11", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_12_Thank_you_both_for_such.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/5280_204996_000056_000000.wav", speaking_style="appreciative and concluding")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_12_Thank_you_both_for_such.wav")))

TTM(text="Upbeat, cheerful closing talk show theme music", length=30, volume=-35, out_wav=os.path.join(wav_path, "fg_music_1_Upbeat_cheerful_closing_talk_show.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_music_1_Upbeat_cheerful_closing_talk_show.wav")))

fg_audio_wavs = []
fg_audio_wavs.append(os.path.join(wav_path, "fg_music_0_Upbeat_inspiring_introductory_talk_show.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_0_Hello_and_welcome_to_the.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_0_Audience_clapping.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_1_Thank_you_Steve_Its_great.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_2_Thanks_Steve_Im_excited_to.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_1_Glass_clinks_lightly_as_guests.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_3_Lets_dive_right_into_it.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_4_Certainly_Spherical_geometry_deals_with.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_2_Audience_clapping.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_5_Thats_quite_fascinating_Mr_Martinez.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_6_Great_question_Steve_In_computer.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_7_Dr_Cheng_youve_mentioned_that.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_8_Of_course_The_study_of.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_3_Audience_murmuring_in_appreciation.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_9_Its_incredible_to_see_how.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_10_As_technology_evolves_were_seeing.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_11_To_add_to_that_I.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_12_Thank_you_both_for_such.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_music_1_Upbeat_cheerful_closing_talk_show.wav"))
CAT(wavs=fg_audio_wavs, out_wav=os.path.join(wav_path, "foreground.wav"))

bg_audio_offsets = []
bg_audio_len = sum(fg_audio_lens[5:18])
bg_audio_offset = sum(fg_audio_lens[:5])
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
