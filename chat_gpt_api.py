import openai
api_key = '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'

def askGPT(text, api_key):
    openai.api_key = api_key
    myQn = """generate 6 more caption given the input like this 
    input- "A man in a hat is displaying pictures next to a skier in a blue hat "
    output - "A man skis past another man displaying paintings in the snow .
              A person wearing skis looking at framed pictures set up in the snow .
    	  A skier looks at framed pictures in the snow next to trees .
              Man on skis looking at artwork for sale in the snow"
    input-"A collage of one person climbing a cliff"          

    output -"A group of people are rock climbing on a rock climbing wall"
            A group of people climbing a rock while one man belays
           Seven climbers are ascending a rock face whilst another man stands holding the rope .
          Several climbers in a row are climbing the rock while the man in red watches and holds the line ".

    input-"""+text+"""
    output-?"""
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=myQn,
        temperature=0.6,
        max_tokens=150,
    )
    ans = response.choices[0].text
    print("more captions are")
    print(ans)

    que = "generate a paragraph given its image captions are as follow"
    que += " " + ans
    print("paragraph is ")
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=que,
        temperature=0.6,
        max_tokens=150,
    )
    print(response.choices[0].text)
def read_api_key(file_path):
    with open(file_path, 'r') as file:
        api_key = file.read().strip()
    return api_key
def main():
    api_key =  read_api_key('api_key.txt')
    text="A black dog runs through the wet grass carrying a green toy in its mouth"
    askGPT(text, api_key)

if __name__ == "__main__":
    main()

