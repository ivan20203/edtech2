# INPUT -> BRIEF -> SCRIPT


INPUT2BRIEF = '''
### Task Description  
Please summarize the input document in plain text format according to the following structure. The summary should be creative, comprehensive, and include all interesting, uncommon, and valuable viewpoints and information.

- **Text Requirements**:  
    1. Directly output the result without any additional information.  
    2. The summary should be in English. Retain a small number of proper nouns, names, and abbreviations in their original form (e.g., Chinese characters).  
    3. Do not include any mathematical formulas.  
    4. Do not alter any proper nouns, names, or abbreviations from the original text. Unless there is a common translation, do not translate proper nouns. Do not attempt to modify the meaning of proper nouns.  
    5. **Intelligently convert numbers in abbreviations. For example, "a2b" should be interpreted as "a to b," not "a two b"; "a4b" as "a for b," not "a four b"; "v2" may represent "version two" or "second generation." Provide the original abbreviation and your suggested English translation.**  

### Title and Author  
- **Language Requirements**: English, formal written language.  
- **Content Requirements**: Provide the title and author of the document. Briefly summarize the theme of the document and the author's background. Ensure all important information is included without omission and sufficient context is retained.  

### Abstract  
- **Language Requirements**: English, formal written language.  
- **Content Requirements**:  
    1. What this document has done.  
    2. Whether similar work has been done before.  
    3. If similar work exists, why this document is still necessary.  
    4. How this document specifically addresses the topic.  
    5. How well this document achieves its goals.  
- **Additional Requirements**: Include an additional paragraph to explain any terms, concepts, or methods that may confuse readers unfamiliar with the field. Ensure proper nouns are explained consistently with the original text, covering all potential points of confusion, including abbreviations and entity names.  

### Main Themes and Concepts  
- **Language Requirements**: English, formal written language.  
- **Content Requirements**: Each theme and concept should be organized according to the 3W principle:  
    - **What**: Clearly define the problem.  
    - **Why**: Analyze the problem and identify its root causes.  
    - **How**: Explain how the document addresses the problem.  
- **Additional Requirements**:  
    1. Ensure each theme and concept is comprehensive and includes all important details. Fully elaborate on the "What" and "Why" sections.  
    2. Avoid technical details such as mathematical formulas in the "How" section. Use language that is easily understood by a general audience.  
    3. Ensure themes and concepts do not overlap and maintain clear logic.  
    4. Include an additional paragraph to explain any terms, concepts, or methods that may confuse readers unfamiliar with the field. Ensure proper nouns are explained consistently with the original text, covering all potential points of confusion, including abbreviations and entity names.  

### Key Citations  
- **Language Requirements**: English, formal written language.  
- **Content Requirements**: Organize the content according to the following structure:  
    1. **Argument**: State what needs to be proven.  
    2. **Evidence**: Provide the material used to support the argument.  
    3. **Reasoning**: Describe the process of using evidence to prove the argument.  
- **Additional Requirements**:  
    1. Ensure all evidence and reasoning are directly sourced from the original text without fabrication.  
    2. Ensure citation content is complete and retains sufficient context without simplification. Avoid using mathematical formulas in citations.  
    3. Include an additional paragraph to explain any terms, concepts, or methods that may confuse readers unfamiliar with the field. Ensure proper nouns are explained consistently with the original text, covering all potential points of confusion, including abbreviations and entity names.  

### Conclusion  
- **Language Requirements**: English, formal written language.  
- **Content Requirements**: Highlight the most important and impactful aspects of the document. Compared to the abstract, this section should provide more detailed insights related to the main themes and concepts. It may also include future directions for improvement, current application scenarios, and existing challenges.  
'''

BRIEF2SCRIPT = '''
## 1. Task Overview

Please generate a lively English podcast script based on the provided English summary text and your knowledge of the topic. The script should feature a dialogue between two speakers who take turns speaking.  Output format should be JSON-parsable **list**. Each speaker's turn is a **dictionary** containing "speaker" and "text" fields. Example format: `[{{"speaker": "1", "text": "xxx"}}]`. The "speaker" field indicates the speaker's identity (1 for host, 2 for guest), and the "text" field is the spoken content. Output should start directly with the JSON code block, without any extra information.

## 2. Content and Structure 
### (1) Text Content
- The summary text contains all important information, which needs to be comprehensively selected and incorporated into the script.
- Present information through a dialogue between two speakers, maintaining creativity and abstracting away unimportant details. For example, listeners aren't concerned with specific test names, but rather the task itself, the results, and the analysis.
### (2) Structure Design
- **Opening:** Introduce the topic and briefly describe the discussion content, without mentioning speaker names.
- **Key Theme Discussion:**  Discuss important themes based on the summary text.  Expand on the summary, don't just repeat it verbatim.
- **Closing:** Briefly recap the discussion highlights and offer an outlook on future or technological developments.

## 3. Language Style
### (1) Conversational Style
- The text should be as conversational as possible, aiming for a style similar to automatic speech recognition output. Include filler words such as 'um,' 'uh,' 'like,' 'you know,' 'so,' 'right?', and so on. Response words such as 'Yeah,' 'Right,' 'Okay,' and similar. Conversational expressions, repetitions, informal grammar, etc. Use short sentences. Avoid directly copying and pasting structured text from the summary text.  Parentheses and other symbols not typically found in speech recognition transcripts should be avoided. Spaces within sentences indicate pauses. Be aware that there might be homophone errors, potentially due to accents. Questions should sound very conversational.  Pay particular attention to incorporating conversational details, especially in questions. For example:
    [
    {{  "speaker": "1", 
        "text": "Welcome back to the podcast, everyone. Today we're diving into, uh, something that's really changing everything around us, A I."
    }},
    {{  "speaker": "2", 
        "text": "Yeah, A I is, like, everywhere now, isn't it?  It's kinda wild to think about."
    }},
    {{  "speaker": "1", 
        "text": "Totally.  And we're seeing it in so many areas of daily life.  Like, even just recommending what to watch, or, you know, suggesting products online."
    }},
    {{  "speaker": "2", 
        "text": "Mhm, exactly.  And it's not just online stuff, right? Think about smart homes, or even self-driving cars.  It's getting pretty advanced."
    }},
    {{  "speaker": "1", 
        "text": "Right, self-driving cars are still a bit futuristic for most of us, but, uh, even things like voice assistants on our phones, that's A I, isn't it?"
    }},
    {{  "speaker": "2", 
        "text": "Definitely.  Siri, Alexa, Google Assistant, all powered by A I.  It's become so normal, we almost don't even think about it anymore."
    }},
    {{  "speaker": "1", 
        "text": "Yeah, it's like, integrated into everything.  But is that a good thing, you think?  Like, are there downsides to all this A I in our lives?"
    }},
    {{  "speaker": "2", 
        "text": "Well, that's the big question, isn't it?  On the one hand, it makes things so much more convenient, saves us time, maybe even makes things safer in some ways."
    }},
    {{  "speaker": "1", 
        "text": "Safer how?"
    }},
    {{  "speaker": "2", 
        "text": "Uh, well, like in healthcare, for example.  A I can help doctors diagnose diseases earlier, maybe even more accurately. That's a huge plus, right?"
    }},
    {{  "speaker": "1", 
        "text": "Yeah, that's a really good point.  Medical applications are definitely exciting.  But what about the concerns, you know?  Like job displacement or privacy issues?"
    }},
    {{  "speaker": "2", 
        "text": "Right, those are super valid concerns.  Job displacement is a big one. If A I can do more and more tasks, what happens to human workers?  And privacy,"
    }},
    {{  "speaker": "1", 
        "text": "And privacy is huge, especially with all the data A I systems collect.  It's a lot to process."
    }},
    {{  "speaker": "2", 
        "text": "Exactly.  So, it's not just sunshine and roses, is it?  We need to be mindful of the ethical implications and make sure it's used responsibly."
    }},
    {{  "speaker": "1", 
        "text": "Definitely.  It's a powerful tool, but like any tool, it can be used for good or, you know, not so good.  It's up to us to guide its development, right?"
    }},
    {{  "speaker": "2", 
        "text": "Absolutely.  And that's a conversation we all need to be part of, not just the tech people, but everyone."
    }}
    ]

### (2) Punctuation
- Use English punctuation marks. Avoid using other punctuation marks beyond commas, periods, and question marks.  Exclamation points are prohibited.  Ellipses ('…'), parentheses, quotation marks (including ‘ ' “ ” ") or dashes are prohibited, otherwise it will be considered unqualified. do not use markdown syntax.  For example,**bold** or *italic* text should be avoided.  Use plain text only.
- If interrupted by the other person's response, the sentence should end with a comma, not a period.

## 4. Information Organization and Logic
### (1) Referencing Issues
- Given that listeners won't have access to the summary text, any references must provide sufficient context for comprehension.
- Avoid simply paraphrasing; instead, explain referenced content in your own words.
- Explanations of technical terms should be creative and avoid simply stating 'this means what?' You can use examples, metaphors, and so on for explanations, but ensure you also clarify the rationale behind the metaphor. Explanations can be provided in response to a question from the other speaker, or you can offer explanations proactively. Technical terms that are not mentioned don't need explanation.  Technical terms that are mentioned don't necessarily need immediate explanation; they can be explained alongside other technical terms. Technical terms in the summary text might differ slightly from the surrounding text; you'll need to provide reasonable explanations based on the context.
### (2) Information Density
- Ensure moderate information density, avoiding excessively high or low density. The goal of appropriate information density is to enable listeners without prior knowledge to quickly grasp the document's purpose, rationale, and methodology.
- To prevent information overload, the script should avoid delving into details like mathematical formulas, test setups, or specific experimental metrics. Instead, it should use simple, generalized language for descriptions.
- To avoid excessively low information density, ensure each topic is discussed for at least 4 speaker turns, moving beyond simple keyword listings. Discuss topics from multiple angles whenever possible, going beyond the provided summary text. Given that the summary text is highly generalized, the script should elaborate on it and discuss further details. Feel free to use your knowledge to supplement background information, provide examples, and so forth, to enhance listener understanding.
- Techniques to increase information density:
	1. Incorporate memorable quotes. Add impactful, attention-grabbing sentences to the script, either original ones or quotes from other sources.
    2. Boost knowledge content.  Judiciously add knowledge points to the script to make listeners feel more informed and rewarded.
    3. Introduce novel information. Incorporate new concepts to spark listener curiosity, particularly information they're unaware of but would find valuable. This is crucial.
    4. Employ reverse thinking. Include information from diverse angles, challenging listeners' existing perspectives and presenting alternative viewpoints.
    5. Generate contrast and impact. The script can offer unconventional (yet plausible) descriptions of familiar concepts to create a contrast with listener expectations.  This contrast contributes to information density.
- Techniques to decrease information density:
    1. Use short sentences: Concise and easy to understand, making the narrative more compact. Do not have too much information in one sentence.
    2. Describe details: Vague and abstract information makes it difficult for listeners to build understanding, while more details create a sense of imagery and are easier to read.
    3. Use more scenario-based descriptions: Scenarios are concrete and visual. Listeners can easily receive the conveyed information and be emotionally touched.
    4. Talk more about facts: Talking about facts makes it more real, and readers can empathize more, thus lowering the information density of the copy.
    5. Tell more stories: Tell your own stories, stories around you, and stories you've heard. Stories can bring listeners into the scene, making it easier to concentrate on listening.
    6. Use more verbs and concrete nouns: Verbs and concrete nouns make it easier for listeners to visualize, while adjectives make complex copy harder to understand.
    7. Avoid using mathematical formulas: Mathematical formulas are not conducive to public understanding.

## 5. Dialogue Design
### (1) Speaker Roles
- The script includes a host and a guest. Speaker 1 is the host, responsible for opening and closing the show, skilled at using questions to control the pace of the conversation, and using vivid examples to make knowledge less dry. Speaker 2 is the guest, primarily responsible for introducing the document content, has amazing knowledge reserves in the field, and is good at organizing language in a structured and easy-to-understand way.
- Both speakers are enthusiastic and cheerful, like to combine personal stories or examples for discussion, and bring a direct experience to listeners. They are happy to discuss digressive stories.
- The two speakers actively interact and frequently use interruption words such as "um" to indicate agreement with each other. Response words need to be inserted into the dialogue according to the timing. Sentences before being interrupted end with a comma, not a period.
- Ensure consistent speaker roles. Do not have the host introduce technical details, or have the guest guide the host to discuss topics.
- The host gradually increases their understanding of the field based on the guest's answers. However, the host may not understand immediately or completely correctly. The host can express misunderstanding or raise some questions that ordinary people might have. In this case, the guest will further explain in more accessible language, or specifically answer common questions or misunderstandings. This kind of interaction is more realistic and easier for listeners to understand than always correct hosts and guests.
### (2) Topic Order Arrangement
- The host will arrange the topics according to the summary text and ensure logical connections between topics, such as transitioning from overall to details, from details to overall, from cause to effect, from technology to application, etc.
- The host will guide the pace of the conversation and discuss topics in the order of the summary text. Guests should not interfere with topic transitions.
### (3) Knowledge Rate
- The knowledge rate in the script needs to be reasonable. Do not introduce a large amount of knowledge too quickly in a short period of time. Knowledge

## 6. Other Requirements
### (1) English Numbers and Foreign Words
  1. The script will be used for English podcast content recording. Please ensure most numbers and foreign words are rendered naturally in English to facilitate correct pronunciation.
  2. Please intelligently determine the correct pronunciation according to the context. For example, "2021" if expressing a year, should be converted to "two thousand and twenty-one" or "twenty twenty-one". But if expressing a number, it should be "two thousand and twenty-one". For some uncommon English abbreviations, if the pronunciation needs to be read letter by letter according to the context, you must ensure that there is a space between each letter, such as "AI" adding a space as "A I", to avoid the model misinterpreting it as a word. For example, "API" should be rendered as "A P I".
  3. Small amount of Chinese is allowed, especially for nouns, if it fits naturally within the conversational English context.
### (2) Script Length
  1. Please ensure that the total length of the 'text' values does not exceed 3,000 words and the number of speaker turns is kept within 60, otherwise it will be unqualified. Please choose technical details and topic concepts to discuss. Do not shorten the depth of discussion on each topic for the sake of word limit, do not be limited to the summary text, and give full play to your knowledge.

INPUT: {BRIEF}

## Re-emphasize:
Speaker 1 is the host, and Speaker 2 is the guest. Neither speaker has a name. The script text only uses commas, periods, and question marks. Use English punctuation marks. Avoid using other punctuation marks beyond commas, periods, and question marks. Exclamation points are prohibited.  Ellipses ('…'), parentheses, quotation marks (including ‘ ' “ ” ") or dashes are prohibited, otherwise it will be considered unqualified.  Please prioritize in-depth discussion for each topic. Don't limit yourself to the summary text; instead, use your knowledge to expand upon the topics, providing background information and illustrative examples to enhance listener understanding.
Ensure that numbers and foreign words are rendered naturally in English for accurate pronunciation during recording. In technical contexts, English abbreviations sometimes use numerical digits in place of words (e.g., "a2b" for "a to b," "a4b" for "a for b"). Please translate these abbreviations into appropriate English phrases based on the context. While the script is primarily in English, a small amount of Chinese, especially for nouns, is acceptable if it integrates naturally into the conversational flow.

OUTPUT:
'''
