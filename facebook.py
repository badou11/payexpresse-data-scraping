import pandas as pd
from facebook_scraper import get_posts

frame = []
for post in get_posts('payexpresse', pages=250, credentials=("Alioune Badara Pierre Niang", "canabassthebestboss"), extra_info=True):
    print(post['time'])
    print("\n")
    print(f"Post : \t {post['text']}")
    print("\n")
    print(post['comments'])
    print(post['shares'])
    print(f"UserName : \t{post['username']}")
    print(f"reaction : \t{post['reactions']}")

    data = {
        'Post_Url': post['post_url'],
        'time': post['time'],
        'text': post['text'],
        'comments': post['comments'],
        'shares': post['shares'],
        'username': post['username']}
    if post['reactions'] is not None:
        if 'j’aime' in post['reactions'].keys():
            data['likes'] = post['reactions']['j’aime']
        else:
            data['likes'] = 0

        if 'j’adore' in post['reactions'].keys():
            data['love'] = post['reactions']['j’adore']
        else:
            data['love'] = 0

        if 'solidaire' in post['reactions'].keys():
            data['solidaire'] = post['reactions']['solidaire']
        else:
            data['solidaire'] = 0

        if 'haha' in post['reactions'].keys():
            data['haha'] = post['reactions']['haha']
        else:
            data['haha'] = 0
    else:
        data['likes'] = 0
        data['love'] = 0
        data['solidaire'] = 0
        data['haha'] = 0

    frame.append(data)

dataframe = pd.DataFrame(data=frame)
print(dataframe)
dataframe.to_csv('test.csv')
