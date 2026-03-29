# KVI win cases (KVI succeeds; GraphRAG fails)

- Source predictions: `experiments/exp01_main_qa/results/multihop_hotpot_n40/predictions.jsonl`
- Cases collected: **8**
- Require RAG fail: **False**

## Case 1 — 5a8b57f25542995d1e6f1371

**Question**: Were Scott Derrickson and Ed Wood of the same nationality?

**Gold**: yes

**EM**: KVI=1, GraphRAG=0, RAG=0

**KVI prediction**: American/yes. Explanation follows:

Analyzing each provided point/document evidence carefully:

1. Confirms that both Woodward & Wood were filmmakers/native Texans/American.
2. -4. Establishes that Film Noir/Film Detective Films/Woodson/Woodside Productions/Scott Derrickson is/was an executive/producers/company/team behind production/distribution/marketing/remakes/revivals/remaking/remade/remaster/music/videos/releases/etc.
5. -10. Specifies that Derrickson-directed/produced/acted-in/more/sci-fi ...

**GraphRAG prediction**: ! entity! no.!

**RAG prediction**: Ed Wood was likely American, given that he owned large plantations in what appears to be the southern United States during the early 20th century.
On the other hand, Scott Derrickson was born in Texas and has lived in places like Los Angeles, indicating he is also American.
However, specific nationalities beyond being from the U. [ungrounded]
S.
are not provided for either individual based on the available information. [ungrounded]

**Gold supporting sentences**:
- Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.
- Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director.

## Case 2 — 5a7bbb64554299042af8f7cc

**Question**: Who is older, Annie Morton or Terry Richardson?

**Gold**: Terry Richardson

**EM**: KVI=1, GraphRAG=0, RAG=1

**KVI prediction**: If not enough information:

The evidence indicates that Annie Morton, born in 1970, is younger than Terry Richardson, who was either born no later than 1965 based on mentions of him being at least 50 years old ("at 50") and active professionally since then without discrepancies suggesting otherwise. 

Terry, elder.: Inferred indirectly. Yes/No question omitted explicitly stating comparison needed. Evidence points conclusively showing age disparity between two public-facing celebrities indicating ...

**GraphRAG prediction**: Yes!! No! Terry! Terry!! 

Annie!! No! Terry" is!! 

T!erry! 

T!erry 

!Terry 

Terry 

No! Terry! 

T!erry!

The correct! answer! is!!! 

T!ERRY! 

T!erry!! 

T!erry! ! 

!T!erry! !

T!erry! !! 

T!erry! !!! 

T!erry!!!!! 

T!erry!!!!!! 

T!erry!! 

T!err!y! 

!!!!T!erry! 

!!!!T!erry! !!!! 

!T!erry" 

!T!erry# 

!T!erry$ 

!T!erry% 

!T!erry& 

!T!erry' 

!T!erry( 

!T!erry) 

!T!erry* 

!T!erry+ 

!T!erry,- 

!T!erry./ 

!T!erry0 

!T!erry1 

!T!erry2 

!T!erry3 

!T!erry4 

!T!erry5 

!T!e ...

**RAG prediction**: Annie Morton was born on October 8, １９７１, while information about Terry Richardson's birth year is not provided in the given evidence.
Therefore, we cannot determine who is older between Annie Morton and Terry Richardson based on the available information.

**Gold supporting sentences**:
- Annie Morton (born October 8, 1970) is an American model born in Pennsylvania.
- She has been photographed by Helmut Newton; Peter Lindbergh; Annie Leibovitz; Richard Avedon; Juergen Teller; Paul Jasmin, Mary Ellen Mark and Terry Richardson, and modeled for Donna Karan, Givenchy, Guerlain, Chanel, "Harper's Bazaar", "Sports Illustrated" and Victoria's Secret.
- Terrence "Uncle Terry" Richardson (born August 14, 1965) is an American fashion and portrait photographer who has shot advertising campaigns for Marc Jacobs, Aldo, Supreme, Sisley, Tom Ford, and Yves Saint Laurent among others.

## Case 3 — 5ab3e45655429976abd1bcd4

**Question**: The Vermont Catamounts men's soccer team currently competes in a conference that was formerly known as what from 1988 to 1996?

**Gold**: the North Atlantic Conference

**EM**: KVI=1, GraphRAG=0, RAG=0

**KVI prediction**: Answer is correct:

The question translates accurately and sufficiently provided textually :

Final entity: 

North Atlantic  

So we could just return : 

Final 
Concise :
The North Atlantic Conference To confirm comprehensiveness, let us analyze logically structured information succinct summaries briefly matched:

Original Text Analysis:
- Mentions key points correctly named entities/entities names/entities classified directly/stated explicitly/only one correct match/topic needed clarified/con ...

**GraphRAG prediction**: ! no
To! yes
The!! answer!!! is!! !the!! Eastern! College! Athletic! Conference! -! North! (ECAC! North! ),!!!!!! since!!!!!"! the!! conference!!! was!!!!!#!!! known!! as! EC! AC! -! North! from!!!! 1979!! to!!! 1 988! and!!! NA! T! C! from!!!"!! 1988!! to!!"!!!!! 996! before!!! becoming!!!!!$!! the!!! current!! name! American! East!! Conference! .!

Therefore!! ,!!! the!!"!! correct!!!!! answer!!"! is!! !no! because!!!!! the!"!!!"!"!! question!!!!! asks!!!!!%! for!!!! its!!!!!&!!!!!'!

**RAG prediction**: 从1988年到1996年，Vermont的Catamounts男子足球队是美国东北部联盟（America East Conference的一员。
该信息主要涉及篮球和冰球团队，因此关于男子足球的具体历史细节不足，但可以确定的是他们在此期间属于America East Conference。

**Gold supporting sentences**:
- The Vermont Catamounts men's soccer team represents the University of Vermont in all NCAA Division I men's college soccer competitions.
- The team competes in the America East Conference.
- The conference was known as the Eastern College Athletic Conference-North from 1979 to 1988 and the North Atlantic Conference from 1988 to 1996.

## Case 4 — 5ab29c24554299449642c932

**Question**: Are Giuseppe Verdi and Ambroise Thomas both Opera composers ?

**Gold**: yes

**EM**: KVI=1, GraphRAG=0, RAG=1

**KVI prediction**: Yes.

Justification/chart/logic/path/story:

1. From basic information available to understand who they each were/is — 
   - [Giuseppe Antonio Maria Verdi / Joseph Wecksbach ](October 1813/1901) : An influential, renowned classical Western male human, creator of many famous musical theatrical performances/games/music/artifacts/frameworks/rules/terms/operator/configuration/orchestra/concert/church/service/community/social/activity/games/videos/books/movies/etc. (Italian opera composer)
   
   ➕ D ...

**GraphRAG prediction**: ! entity!!!! no! !!!!

**RAG prediction**: Yes, Giuseppe Verdi was indeed an opera composer, known for his works such as "La Traviata" and "Aida.
" While there isn't direct information about Ambroise Thomas' profession from these details, it can be inferred that since this set of evidence focuses on Giuseppe Verdi and does not mention Ambroise Thomas, we would need additional information to confirm if Ambroise Thomas was also an opera composer.

**Gold supporting sentences**:
- Giuseppe Fortunino Francesco Verdi (] ; 9 or 10 October 1813 – 27 January 1901) was an Italian opera composer.
- Charles Louis Ambroise Thomas (5 August 1811 – 12 February 1896) was a French composer, best known for his operas "Mignon" (1866) and "Hamlet" (1868, after Shakespeare) and as Director of the Conservatoire de Paris from 1871 till his death.

## Case 5 — 5ae2070a5542994d89d5b313

**Question**: Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice?

**Gold**: Badly Drawn Boy

**EM**: KVI=1, GraphRAG=0, RAG=1

**KVI prediction**: If not feasible:

phrase).

Explanation/Ideas/Resources/Steps):

1. From the evidence provided, we can ascertain that the band/group in question for comparison regarding a potential higher instrument-to-person ratio is "Badly Drawn Boy."
    - Evidence points indicate "Tom Rothrock," presumably a music mogul/producers' liaison, collaborates extensively across numerous high-profile indie-rock/pop acts/groups/artists internationally but notably composes orchestrations/soundtracks extensively showc ...

**GraphRAG prediction**: ! entity,!! no,!!
To!!! compare!!!!! the!!!!!!"instrument!to!!person!ratio!for!!!!Bad!!Draw!n!Boy!and!!!!!#Wolf!!Alice!,"!!!!!$!we!!need!!to!!analyze!!!the!!!composition!!of!!each!!!!act!!and!!!"!!!!!their!!!musical!!!!!%formation!."!For!!!!!Bad"!!Draw!n"!!Boy!,$!it!!is!!!des!!cribed!!!!as!!a!single!artist!or!a!small!ensemble!!with!!multiple!!instruments!!played!!by!!one!!individual!($D!amon!Gough!)$!who!!!also!!writes!and!!produces!music!alone!or!!!!!with!!!!!&support!!!!from!!!!other!!!

**RAG prediction**: 根据提供的信息，无法直接比较Badly Drawn Boy和Wolf Alice的乐器与人员比例。 [ungrounded]
关于Badly Drawn Boy的信息主要涉及其音乐类型，而Wolf Alice则被描述为一个四人组成的另类摇滚乐队。 [ungrounded]

**Gold supporting sentences**:
- Damon Michael Gough (born 2 October 1969, in Dunstable, Bedfordshire), known by the stage name Badly Drawn Boy, is an English indie singer-songwriter and multi-instrumentalist.
- Its members since 2012 are Ellie Rowsell (vocals, guitar), Joff Oddie (guitars, vocals), Theo Ellis (bass), and Joel Amey (drums, vocals).

## Case 6 — 5abd259d55429924427fcf1a

**Question**: Are both Dictyosperma, and Huernia described as a genus?

**Gold**: yes

**EM**: KVI=1, GraphRAG=0, RAG=0

**KVI prediction**: Yes/shortened explanation:Hypertextual relationships indicate:

Both entities refer to botanical taxa classified under the broader 'Huerta' or garden cultivation contextsually implied, however specific descriptions align - 
 Regel et al., 1810 denotes binomial nomenclature chronological parity for respective cactus families/subfamilies designations under Apocynaceae with Apo-cadmus-Scyelepedioidesevolutionary timelines suggesting morphological classifications inherently encompassing dichotomous  ...

**GraphRAG prediction**: ! entity!! no! !

**RAG prediction**: Dictyosperma is indeed described as a genus, specifically a monotypic genus containing only one species, D.
album.
Huernia, on the other hand, also describes a genus, characterized by their specific botanical traits such as bilobed flowers that range from pink to pale violet, and they are endemic to Gibraltar.
Both genera belong to different families and have distinct characteristics.

**Gold supporting sentences**:
- Dictyosperma is a monotypic genus of flowering plant in the palm family found in the Mascarene Islands in the Indian Ocean (Mauritius, Réunion and Rodrigues).
- The genus Huernia (family Apocynaceae, subfamily Asclepiadoideae) consists of stem succulents from Eastern and Southern Africa, first described as a genus in 1810.

## Case 7 — 5a828c8355429966c78a6a50

**Question**: Kaiser Ventures corporation was founded by an American industrialist who became known as the father of modern American shipbuilding?

**Gold**: Henry J. Kaiser

**EM**: KVI=1, GraphRAG=0, RAG=1

**KVI prediction**: Explanation:

Ontario, California - yes

Kaiser Ventures, formerly Kaiser Steel, headquarters located in Ontario, Southern California, was initiated by the renowned entrepreneur and pioneer of maritime engineering, Henry J. Kaiser, embodying quintessential contributions towards transformative advancements within burgeoning naval infrastructures and allied sectors including cementation endeavors through diversified conglomerates bequeathing monumental legacies synonymous perpetually cherished pro ...

**GraphRAG prediction**: No need for![](http://some_image_url.png).
no
根据![](https://v.tiktok.com/l/dQXGZLxhjzr/)提供的![](![](http://some_imgur_url.png))信息![](![](http://%F0%9F%A4%96.com/images/emoji/unicode/1f52c.png))，![](https://v.kuaishou.com/n![](http://some_kshero![](http://some_gif_url.gif)_url.png))可以![](https://v.douyin.com/n![](http:///images/emoji/unicode/![](http://some_emoji![](http://some_toutiao![](http://some_video![](http://some_bilibili![](http://some_weibo![](http://some_sina![](http://some_q![](http:// ...

**RAG prediction**: Henry J.
Kaiser, the founder of Kaiser Ventures Corporation, was indeed known as the father of现代美国船坞工业。

**Gold supporting sentences**:
- Kaiser Ventures (formerly Kaiser Steel) corporation, headquartered in Ontario, California.
- It was founded by Henry J. Kaiser to provide steel plate for the Pacific Coast shipbuilding industry, which expanded during World War II, then shrank, then expanded again during the Korean War.
- Henry John Kaiser (May 9, 1882 – August 24, 1967) was an American industrialist who became known as the father of modern American shipbuilding.

## Case 8 — 5ae4a3265542995ad6573de5

**Question**: Hayden is a singer-songwriter from Canada, but where does Buck-Tick hail from?

**Gold**: Fujioka, Gunma

**EM**: KVI=1, GraphRAG=0, RAG=0

**KVI prediction**: Scroll up/upward/to zenith/piñata/entity/entity/previous/topic/name/title/address/name/title/topic/title/topic/url:

Evidence shows that Haydensinger is from Thornhill, Ontario, while:

Buck-Tick originates from Fujioka, Gunma, Japan.

Final : Japan/topic/title/name/title/topic/title/entities/entities/entities/entities/entities/entities/entity/title/topic/title/topic/topic/topic/topic/topic/topic/title/topic/topic/topic/title/topic/title/topic/topic/title/topic/topic/title/locale/locale/locale/l ...

**GraphRAG prediction**: Japan! Yes!!

**RAG prediction**: 关于Hayden的信息主要涉及他的音乐作品和居住地，与Buck-Tick无关。 [ungrounded]

**Gold supporting sentences**:
- Paul Hayden Desser (born February 12, 1971) who records as Hayden, is a Canadian singer-songwriter from Thornhill, Ontario.
- Buck-Tick (stylized as BUCK-TICK) is a Japanese rock band, formed in Fujioka, Gunma in 1983.

