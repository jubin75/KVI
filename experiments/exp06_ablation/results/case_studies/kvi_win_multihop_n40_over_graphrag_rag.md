# KVI win cases (KVI succeeds; GraphRAG fails)

- Source predictions: `experiments/exp01_main_qa/results/multihop_hotpot_n40/predictions.jsonl`
- Cases collected: **4**
- Require RAG fail: **True**

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

## Case 2 — 5ab3e45655429976abd1bcd4

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

## Case 3 — 5abd259d55429924427fcf1a

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

## Case 4 — 5ae4a3265542995ad6573de5

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

