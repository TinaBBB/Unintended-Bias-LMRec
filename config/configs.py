confidence_interval_dict = {
    0.85: 1.440,
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576
}

city_list = ['Toronto', 'Boston', 'Atlanta', 'Austin', 'Columbus', 'Orlando', 'Portland']

# unique category list here
replacements = {'/': ',',
                ' ,': ',',
                ', ': ','}

# applies for all cities to exclude as people's names
exclusion_list = ['escargot', 'castagna', 'egg', 'patty', 'roadhouse', 'lightner', 'pigeon', 'pierce', 'karseki', 'takumi', 'bon', 'caramel', 'mood', 'martini', 'superb', 'appetizer', 'little', 'tail','service', 'krs', 'capa', 'ich', 'für','cesar', 'landing',
                 'right', 'spiderman', 'right', 'shamu', 'rode','parks','shamu', 'diagon','parks','ist','room','right','jurassic','popeye','forbidden','fun','bei', 'seaworld','rose','waldorf','area','alley','kong','ride',
                 'hunnel','land','world','die','walk', 'Ich', 'dumplings', 'crab','santo','ceasar','cap', 'fantastic','soo', 'del','law','strip', 'caesar', 'duck','filet','broccoli','crabb','dumpling','bone','grille','bartender','seared',
                 'sorellina','','mooo', 'gyutoro','bull', 'grilled', 'driskill', 'grilled','nigiri', 'yellowtail','groupon','wonderful','sullvian','dine','son', 'steakie','maki', 'tako', 'qui', 'yokai','hama', 'wagyu','machi', 'uchiko', 'chili','chef','cure', 'austin', 'brûlée', 'provided','watering','deserves','crosstown','','brule', 'toca', 'pear', 'kaji','classy','shoushin','ritz','thon','alo','ahi','choy','wise','nyc','cotta','bok','curry','word','decor','warm','stuffy','banana','handroll','mind','zen','food',
                 'huge','blanc','fan','wallet','panna', 'roe', 'yaki','fois','berry','bream','ricotta','sea', 'tartare', 'gras','uni','saki','benedit', 'benedict','lemon','herb', 'simcoe','cracker','omakase','flavourful','riotta',
                 'harbour','carlton', 'hen','grapefruit', 'scarborough','scallop','sashimi','foie', 'grill', 'kitchen', 'burger', 'market', 'cafe', 'thai', 'poke', 'bros', 'columbus', 'café', 'applebee', 'dirth', 'bar',
                 'steak', 'cheese', 'house', 'caffè', 'cake', 'bay', 'bistro', 'shabu', 'pizzeria', 'caffe', 'boston', 'back', 'la', 'de',
                 'brasserie', 'brewer', 'kingfish', 'il', 'ristorante', 'mastro', 'chão', 'caffé', 'yakitori','wine','sushi','restobar',
                 'bonapita', 'bangkok', 'izakaya', 'burrito', 'ciao', 'toro', 'tratoria', 'mexicano', 'north', 'ramen', 'kington', 'pho', 'kingka',
                 'kabob', 'markham', 'tetsu', 'tofu', 'chocolatemaker', 'asian', 'cacao', 'creperie', 'bloor', 'noodle', 'york', 'kinton', 'yorkdale',
                 'santouka', 'eglinton', 'soon', 'cuisine', 'danforth', 'katsu', 'toronto', 'jabistro', 'sake', 'sabai', 'kenzo', 'pai', 'katsuya', 'kinka', 'hokkaido',
                 'bake', 'ni', 'ne', 'northen', 'works', 'tai', 'espresso', "l\'espresso", 'trattoria', 'code', 'golden', 'bbq', 'cluck', 'northern', 'sélect', 'pot',
                 'hot', 'mercurio', 'beef', 'shoryuken', 'brew', 'sooncook', 'caribbean', 'tea', 'buddha', 'du', 'hy', 'yu', 'simmer', 'vaughn', 'pattiserie', 'bene','eat',
                 'yasu','poutini', 'canada', 'sneaky', 'bannock', 'fat', 'fried', 'dessert', 'chai', 'sneaky', 'ok', 'mother', 'soups', 'jerk', 'moose',  'kaka', 'dundas',
                 'angus', 'gong', 'cha', 'momo', 'mexicana', 'wilbur', 'pablo','boulud', 'edulis','wah', 'cheesetart', 'yuzu', 'miku', 'destileria', 'rice','springs',
                 'mediterranean', 'doc', 'latina', 'taqueria', 'springs', 'mcdonald', 'steakhouse', 'knight','bagels','italiano', 'uncle', 'newk', 'bangladeshi','cook',
                 'chin', 'papi', 'larger', 'gourmet', 'staplehouse', 'steakbar', 'mac', 'kr', 'baby', 'delux', 'larger', 'hawaiian', 'lazy', 'beach','eastlake', 'vegan', 'taco',
                 'indian','hai', 'divine', 'mama', 'barbecue', 'hill','ranch','young','lane','street','clay','fleming','sushiko','deli','coffee', 'south','temps', 'nous',
                 'wahoo','orlando','chinese','paris','hunter','bubbalou','habibi','hamburger','italiana','garden','tango','deep','seafood','chop','panini','bakery',
                 'bubble','patisserie','tacos','bubba','soupa','shrimp','buffett','meatball','kabooki','artisan','bahama','wolf','french','narcoossee','frog','boxer','pdx',
                 'pbj','teriyaki','broder','siri','man','bento','osakaya','oyster','grant','brewing','donuts','crêperie','st','great','mushroom','said','rosemary','gravy','bacon','strawberry','care','dog','tuna','melt','crosshairs','booth','grabbed',
                  'rude','kinda','waitress','said','salmon','bit','pricey','pizzatna','friend','away','bomb','roll','miles','runky','mustard','sweet',
                  'jesus','feeling','beard','beard','holly','cow','picky','drinker','creamer','soy','mouth','heaven','gravy','son','veryyyy','strong','pure',
                  'ate','fluffy', 'yummy','tartar','bun','mayo','lunch','wold','pie','dry','salmon','hash', 'yum',
                  'boyfriend','regular','msg','biggie','halo','carrot','salsa','killer','circle',
                  'like','delivers','pizza','basil','team','awesome', 'smokey','honey','morning','done',
                  'beau','mains','app','guy','park','creme','amazing','cowboy','brulee','bit','said','nice','spicy',
                      'low','butter','chocolate','waiter','top','mouth','brúlee','rich','ribeye','felt','server','bill','attentive','husband','crême','got','good',
                      'dominic','char','salad','pink','took','carrot','salmon','team','cousin','refectory','till','came','lobster','mussel','vanilla','crême','thing','waitress',
                      'brûlêe','knowledgeable', 'crème','best','bacon','juicy', 'right'
                      'Owner','Pancake','Yogourt','Doughnuts']

nightlife_list = {
        'arcades', 'bars', 'bar crawl', 'beer', 'beer bar', 'brewpubs', 'cabaret',
        'dance clubs', 'champagne bars', 'cocktail bars', 'dance clubs', 'dive bars', 'gastropubs',
        'gay bars', 'hookah bars', 'irish pub', 'izakaya', 'jazz & blues', 'karaoke', 'lounges',
        'pool halls', 'pool & billiards', 'music venues', 'nightlife', 'party supplies', 'piano bars', 'pubs',
        'recreation centers', 'sports bars', 'sports clubs', 'tabletop games', 'tapas bars',
        'tiki bars', 'whiskey bars', 'wine & spirits', 'wine bars'}