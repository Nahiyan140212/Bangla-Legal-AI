#Keeping only the bengali and english word
def basic_text_pre(text):
    text = re.sub(r'[\n\r\t]', ' ', text)
    text = re.sub(r'[^a-zA-Z\u0980-\u09FF\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text.strip()

#If we remove the english words we could lose the important context from the text.So instead of removing the english word
#we replace the english word with thier corresponding Bengali meaning
#At first we will create the dictionary
Question = ' '.join(
    df['Question (Bengali)']
    .dropna()
    .astype(str)
    .map(str.strip)
    .loc[lambda x: x != '']
)

Answer = ' '.join(
    df['Answer (Bengali)']
    .dropna()
    .astype(str)
    .map(str.strip)
    .loc[lambda x: x != '']
)
all_text = (Question) + ' ' + (Answer)
all_text = all_text = basic_text_pre(all_text)
#Finding all the english words
english_words = re.findall(r'\b[a-zA-Z0-9]+\b', all_text)
# Mapping from English digits to Bengali digits as it will not convert via Google Translator
eng_to_bangla_digit_map = str.maketrans('0123456789', '০১২৩৪৫৬৭৮৯')

def convert_english_digits_to_bangla(text):
    return text.translate(eng_to_bangla_digit_map)
english_words= english_words = [convert_english_digits_to_bangla(word) for word in english_words]
# Create the dictionary
english_to_bangla = {}
english_words = set(english_words)
# Translate each word and store in dictionary
for word in english_words:
    try:
        bangla = GoogleTranslator(source='en', target='bn').translate(word)
        english_to_bangla[word] = bangla
    except Exception as e:
        english_to_bangla[word] = word
        print(f"Error translating '{word}': {e}")
#As some words did not translate properly we manually update the dictionary
english_to_bangla['nid'] = 'পরিচয়পত্র'
english_to_bangla['rs'] = 'আরএস'
english_to_bangla['ca'] = 'সিএ'
english_to_bangla['sa'] = 'এসএ'

#replaceing the english word to its corresponding bengali meaning
def replace_english_with_bangla(text, translation_dict):
    words = text.split()
    replaced_words = [translation_dict.get(word, word) for word in words]
    return ' '.join(replaced_words)

def tokenize_text(text):
    return word_tokenize(text)
def remove_stopwords(tokens):
    new_words =  [word for word in tokens if word not in stop_words]
    return ' '.join(new_words)

def Text_Preprocessing(text):
  if not isinstance(text, str):
      return ''
  text = basic_text_pre(text) # Basic preprocessing
  text = replace_english_with_bangla(text,english_to_bangla) # Converting english word into its Bengali word
  text = tokenize_text(text) # Tokenize at word level
  text = remove_stopwords(text) # Stop words removal
  return text
