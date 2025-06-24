
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

TTS(text="Ladies and gentlemen, welcome to today's episode of 'Talk Show Night'! We'll be diving into a very intriguing topic: ‘What are the primary factors that influence redditor behavior?’ We have two eminent guests joining us. First let me introduce you to Dr. Emily Rosen, a well-respected social psychologist with extensive research on digital behavior.", speaker_id="85", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_0_Ladies_and_gentlemen_welcome_to.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/8307_120458_000034_000001.wav", speaking_style="Excited and energetic")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_0_Ladies_and_gentlemen_welcome_to.wav")))

TTA(text="Audience clapping", length=3, volume=-35, out_wav=os.path.join(wav_path, "fg_sound_effect_0_Audience_clapping.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_0_Audience_clapping.wav")))

TTS(text="Thank you for having me. I'm excited to delve into the factors that shape behavior on Reddit, from community-driven norms to the platform's unique features.", speaker_id="132", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_1_Thank_you_for_having_me.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7078_271888_000042_000000.wav", speaking_style="Polite and knowledgeable")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_1_Thank_you_for_having_me.wav")))

TTS(text="We're thrilled to have you. And let's also welcome our second guest, Mr. Samuel Hughes, a data scientist at Reddit who deals with user behavioral data firsthand!", speaker_id="85", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_2_Were_thrilled_to_have_you.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/8307_120458_000034_000001.wav", speaking_style="Animated and enthusiastic")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_2_Were_thrilled_to_have_you.wav")))

TTA(text="Glass clinks lightly as guest settle in", length=2, volume=-35, out_wav=os.path.join(wav_path, "fg_sound_effect_1_Glass_clinks_lightly_as_guest.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_1_Glass_clinks_lightly_as_guest.wav")))

TTS(text="Thanks for having me on the show! Reddit's user behavior is a fascinating subject, and I look forward to sharing some insights from a data perspective.", speaker_id="217", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_3_Thanks_for_having_me_on.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/66_354_000019_000003.wav", speaking_style="Grateful and excited")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_3_Thanks_for_having_me_on.wav")))

TTS(text="To kick things off, Dr. Rosen, could you guide us through the primary factors that influence redditor behaviors?", speaker_id="85", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_4_To_kick_things_off_Dr.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/8307_120458_000034_000001.wav", speaking_style="Enthusiastic and engaging")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_4_To_kick_things_off_Dr.wav")))

TTA(text="Audience clapping", length=2, volume=-35, out_wav=os.path.join(wav_path, "fg_sound_effect_2_Audience_clapping.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_sound_effect_2_Audience_clapping.wav")))

TTS(text="Certainly. Some of the major elements include the community influence, the platform's design, and its anonymous nature. The hive-mind mentality fostered by the upvote-downvote system, for instance, greatly impacts content and discussions. Also, each subreddit has its own unique set of norms which shape user behaviors.", speaker_id="132", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_5_Certainly_Some_of_the_major.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7078_271888_000042_000000.wav", speaking_style="Informed and clear")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_5_Certainly_Some_of_the_major.wav")))

TTS(text="Fascinating! How about you, Samuel? From a data science perspective, what would you say impacts user behavior?", speaker_id="85", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_6_Fascinating_How_about_you_Samuel.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/8307_120458_000034_000001.wav", speaking_style="Inquisitive and engaging")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_6_Fascinating_How_about_you_Samuel.wav")))

TTS(text="Well, echoing Dr. Rosen's points, the community or subreddit that users choose to be a part of heavily influences their behavior. The content and thread that foster discussion also play a vital role in this aspect.", speaker_id="217", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_7_Well_echoing_Dr_Rosens_points.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/66_354_000019_000003.wav", speaking_style="Confident and concise")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_7_Well_echoing_Dr_Rosens_points.wav")))

TTS(text="Dr. Rosen, what are some behavioral tendencies unique to Redditors compared to other social media users?", speaker_id="85", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_8_Dr_Rosen_what_are_some.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/8307_120458_000034_000001.wav", speaking_style="Engaged and curious")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_8_Dr_Rosen_what_are_some.wav")))

TTS(text="Reddit users generally appreciate meaningful content. Posts that showcase research, thoughtfulness or unique experiences usually receive high levels of interaction. Also, redditors blend humor and seriousness in their posts and comments. Plus, there's a significant element of supportiveness seen in many communities.", speaker_id="132", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_9_Reddit_users_generally_appreciate_meaningful.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7078_271888_000042_000000.wav", speaking_style="Explanatory and engaged")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_9_Reddit_users_generally_appreciate_meaningful.wav")))

TTS(text="Samuel, could you weigh in on this aspect?", speaker_id="85", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_10_Samuel_could_you_weigh_in.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/8307_120458_000034_000001.wav", speaking_style="Curious and encouraging")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_10_Samuel_could_you_weigh_in.wav")))

TTS(text="Absolutely. Redditors often engage in deep, specific fields, and communal learning - something unique to Reddit’s structure. Another behavioral aspect worth mentioning is the use of 'throwaway' accounts for discussing sensitive or controversial topics.", speaker_id="217", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_11_Absolutely_Redditors_often_engage_in.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/66_354_000019_000003.wav", speaking_style="Insightful and clear")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_11_Absolutely_Redditors_often_engage_in.wav")))

TTS(text="Indeed, the anonymous nature of Reddit seems to significantly impact user behaviors. Dr. Rosen, could you elaborate?", speaker_id="85", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_12_Indeed_the_anonymous_nature_of.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/8307_120458_000034_000001.wav", speaking_style="Facilitative and curious")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_12_Indeed_the_anonymous_nature_of.wav")))

TTS(text="Well, this anonymity provides users the freedom to voice personal or controversial views without fear of judgment. While this fosters diverse discussions, it can also instigate less empathic or decent behaviors.", speaker_id="132", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_13_Well_this_anonymity_provides_users.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7078_271888_000042_000000.wav", speaking_style="Detailed and engaging")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_13_Well_this_anonymity_provides_users.wav")))

TTS(text="Samuel, any thoughts?", speaker_id="85", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_14_Samuel_any_thoughts.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/8307_120458_000034_000001.wav", speaking_style="Curious and facilitative")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_14_Samuel_any_thoughts.wav")))

TTS(text="Anonymity on Reddit indeed encourages diverse dialogues. However, as Dr. Rosen mentioned, it can also lead to the rise of harmful behaviors which needs to be checked.", speaker_id="217", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_15_Anonymity_on_Reddit_indeed_encourages.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/66_354_000019_000003.wav", speaking_style="Affirmatory and thoughtful")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_15_Anonymity_on_Reddit_indeed_encourages.wav")))

TTS(text="Now, let's talk about fostering positive behavior on Reddit. Can you talk us through this Dr. Rosen?", speaker_id="85", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_16_Now_lets_talk_about_fostering.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/8307_120458_000034_000001.wav", speaking_style="Inviting and positive")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_16_Now_lets_talk_about_fostering.wav")))

TTS(text="Reddit enforces clear site rules and relies heavily on moderators to manage each subreddit's norms. To foster positive interactions, I believe they could expand on the current Karma system to highlight good digital citizenship and promote more positive discussions and support.", speaker_id="132", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_17_Reddit_enforces_clear_site_rules.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/7078_271888_000042_000000.wav", speaking_style="Analytical and suggestive")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_17_Reddit_enforces_clear_site_rules.wav")))

TTS(text="And what's your take on this, Samuel?", speaker_id="85", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_18_And_whats_your_take_on.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/8307_120458_000034_000001.wav", speaking_style="Engaging and inquisitive")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_18_And_whats_your_take_on.wav")))

TTS(text="Reddit uses both human moderation and machine learning algorithms to enforce rules and take actions against inappropriate content. But yes, as Dr. Rosen suggested, enhancing user verifications could discourage trolling and incivility, thus fostering a more encouraging and open environment.", speaker_id="217", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_19_Reddit_uses_both_human_moderation.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/66_354_000019_000003.wav", speaking_style="Agreeable and thoughtful")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_19_Reddit_uses_both_human_moderation.wav")))

TTS(text="Thank you both for your incredibly insightful conversation today. Dear audience, we hope you found this discussion engaging and enlightening. Enjoy the rest of your evening!", speaker_id="85", volume=-15, out_wav=os.path.join(wav_path, "fg_speech_20_Thank_you_both_for_your.wav"), speaker_path="/home/ivan/github-repos/edtech/edtech/PodAgent/data/voice_presets_cv_en/ref_wav/8307_120458_000034_000001.wav", speaking_style="Appreciative and charismatic")
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_speech_20_Thank_you_both_for_your.wav")))

TTM(text="Upbeat, cheerful closing talk show theme music", length=30, volume=-35, out_wav=os.path.join(wav_path, "fg_music_1_Upbeat_cheerful_closing_talk_show.wav"))
fg_audio_lens.append(COMPUTE_LEN(os.path.join(wav_path, "fg_music_1_Upbeat_cheerful_closing_talk_show.wav")))

fg_audio_wavs = []
fg_audio_wavs.append(os.path.join(wav_path, "fg_music_0_Upbeat_inspiring_introductory_talk_show.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_0_Ladies_and_gentlemen_welcome_to.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_0_Audience_clapping.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_1_Thank_you_for_having_me.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_2_Were_thrilled_to_have_you.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_1_Glass_clinks_lightly_as_guest.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_3_Thanks_for_having_me_on.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_4_To_kick_things_off_Dr.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_sound_effect_2_Audience_clapping.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_5_Certainly_Some_of_the_major.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_6_Fascinating_How_about_you_Samuel.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_7_Well_echoing_Dr_Rosens_points.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_8_Dr_Rosen_what_are_some.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_9_Reddit_users_generally_appreciate_meaningful.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_10_Samuel_could_you_weigh_in.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_11_Absolutely_Redditors_often_engage_in.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_12_Indeed_the_anonymous_nature_of.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_13_Well_this_anonymity_provides_users.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_14_Samuel_any_thoughts.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_15_Anonymity_on_Reddit_indeed_encourages.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_16_Now_lets_talk_about_fostering.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_17_Reddit_enforces_clear_site_rules.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_18_And_whats_your_take_on.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_19_Reddit_uses_both_human_moderation.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_speech_20_Thank_you_both_for_your.wav"))
fg_audio_wavs.append(os.path.join(wav_path, "fg_music_1_Upbeat_cheerful_closing_talk_show.wav"))
CAT(wavs=fg_audio_wavs, out_wav=os.path.join(wav_path, "foreground.wav"))

bg_audio_offsets = []
bg_audio_len = sum(fg_audio_lens[4:25])
bg_audio_offset = sum(fg_audio_lens[:4])
TTA(text="Soft generic talk show background music", volume=-35, length=bg_audio_len, out_wav=os.path.join(wav_path, "bg_sound_effect_0_Soft_generic_talk_show_background.wav"))
bg_audio_offsets.append(bg_audio_offset)

bg_audio_len = sum(fg_audio_lens[10:25])
bg_audio_offset = sum(fg_audio_lens[:10])
TTA(text="Subtle background noise of rustling paper and pencil scribbling", volume=-35, length=bg_audio_len, out_wav=os.path.join(wav_path, "bg_sound_effect_1_Subtle_background_noise_of_rustling.wav"))
bg_audio_offsets.append(bg_audio_offset)

bg_audio_wavs = []
bg_audio_wavs.append(os.path.join(wav_path, "bg_sound_effect_0_Soft_generic_talk_show_background.wav"))
bg_audio_wavs.append(os.path.join(wav_path, "bg_sound_effect_1_Subtle_background_noise_of_rustling.wav"))
bg_audio_wav_offset_pairs = list(zip(bg_audio_wavs, bg_audio_offsets))
bg_audio_wav_offset_pairs.append((os.path.join(wav_path, "foreground.wav"), 0))
MIX(wavs=bg_audio_wav_offset_pairs, out_wav=os.path.join(wav_path, "res_test.wav"))
