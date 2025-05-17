import pandas as pd

# Create fake news data
fake_news_data = [
    {
        "title": "Biden Secretly Planning to Hand Over US Military Control to United Nations, Sources Say",
        "text": "Sources close to the administration reveal that President Biden has been meeting with UN officials to draft a plan that would place US Armed Forces under direct UN control by 2025. \"This is part of the globalist agenda they've been planning for decades,\" said a White House insider who requested anonymity. \"The documents are being prepared in secret meetings at Camp David.\" Military officials are reportedly furious after learning about the plan, with several generals threatening to resign. \"This is a complete surrender of American sovereignty,\" one official reportedly said. Conservative lawmakers have vowed to fight the move, with Rep. Jim Jordan tweeting: \"First they came for our guns, now they're giving away our military. America FIRST!\". The White House has not responded to requests for comment on what would be the most significant change to US military authority in history.",
        "subject": "News",
        "date": "May 2, 2024"
    },
    {
        "title": "SHOCKING: Leaked Emails Show Fauci Admitted Covid Vaccines Contain Mind-Control Technology",
        "text": "In a bombshell revelation that the mainstream media refuses to cover, recently leaked emails show Dr. Anthony Fauci privately admitted that COVID-19 vaccines contain experimental nano-technology designed to make recipients more compliant to government messaging. \"The microchips are smaller than we thought possible,\" Fauci allegedly wrote to pharmaceutical executives in an email dated March 2021. \"Preliminary results show a 67% increase in acceptance of government narratives after the second dose.\" Several whistleblowers from within the CDC have confirmed the authenticity of these emails, saying they were instructed to hide this information from the public. \"We were told this was for the greater good,\" one CDC scientist said. \"But people deserve to know what's being injected into their bodies.\" Freedom advocates are demanding congressional hearings, while social media platforms have already begun removing posts mentioning the leaked documents.",
        "subject": "News",
        "date": "May 3, 2024"
    },
    {
        "title": "Kamala Harris Caught on Hot Mic: 'Americans Are Too Stupid to Know What's Good for Them'",
        "text": "Vice President Kamala Harris was caught on a hot microphone making disparaging comments about American voters during a closed-door fundraiser with elite donors in San Francisco. \"Most Americans are just too stupid to understand what's good for them,\" Harris can be heard saying in the leaked audio. \"That's why we need to make decisions for them.\" The embarrassing gaffe occurred when Harris thought recording equipment had been turned off following her official remarks. Several attendees appeared uncomfortable as Harris continued, \"If they knew our actual agenda, they'd never vote for us.\" Republican leaders are demanding an apology, with Senator Ted Cruz tweeting, \"Finally the truth comes out about what Democrats really think of everyday Americans.\" The White House has claimed the audio was \"deceptively edited\" but has not released the full recording despite multiple requests from media outlets.",
        "subject": "News",
        "date": "April 29, 2024"
    },
    {
        "title": "Scientists Confirm Global Warming Data Was Fabricated to Secure Research Funding",
        "text": "A shocking admission from leading climate scientists reveals decades of temperature data were deliberately manipulated to create the illusion of global warming. Dr. Henrik Svenson, former director of the Global Climate Research Institute, admitted in a sworn affidavit that his team altered historical temperature records to secure billions in research grants. \"We needed the funding to continue our work, and alarming predictions opened government wallets,\" Svenson stated. The confession has sent shockwaves through the scientific community, with many researchers now calling for a complete audit of all climate research from the past 30 years. Energy companies are reportedly preparing lawsuits against environmental groups that used the fabricated data to block pipeline projects. \"This is the greatest scientific fraud in history,\" said energy analyst Thomas Reynolds. \"Trillions of dollars have been wasted fighting a problem that doesn't exist.\"",
        "subject": "News",
        "date": "May 5, 2024"
    },
    {
        "title": "Trump Files Secret Lawsuit to Prove 2020 Election Was Stolen by Time-Traveling Democrats",
        "text": "Former President Donald Trump has filed a bombshell lawsuit under seal in federal court claiming to have evidence that Democratic operatives used experimental time-travel technology to alter the 2020 election results. According to sources familiar with the filing, Trump's legal team has obtained classified Pentagon documents showing the existence of a top-secret program codenamed \"CHRONOS\" that allows users to make small changes to past events. \"This explains why no conventional fraud was detected,\" said Sidney Powell, who is reportedly advising on the case. \"They literally changed history.\" The lawsuit alleges that Barack Obama authorized the program in 2016 as his \"insurance policy\" against a Trump presidency. Several quantum physicists have reportedly signed affidavits supporting the technical possibility of the claims. Trump supporters are calling this the \"smoking gun\" they've been waiting for, while Democrats dismiss it as science fiction.",
        "subject": "News",
        "date": "April 27, 2024"
    },
    {
        "title": "BREAKING: AOC Proposes Bill to Ban Meat Consumption and Private Car Ownership by 2026",
        "text": "Congresswoman Alexandria Ocasio-Cortez introduced shocking new legislation yesterday that would completely ban meat consumption and private car ownership in the United States by 2026. The \"Green Future Act\" would implement a gradual phase-out of animal products over the next two years, with heavy fines for anyone caught eating or selling meat after the deadline. \"Climate change requires extreme measures,\" Ocasio-Cortez said during a press conference. \"Americans must sacrifice their hamburgers and SUVs if we want to survive.\" The bill also contains provisions for government officials to conduct random home inspections to ensure compliance with the new dietary restrictions. Republican lawmakers immediately condemned the proposal, with House Speaker Mike Johnson calling it \"communist tyranny disguised as environmentalism.\" Several Democrat moderates have privately expressed concern about the bill's radical approach, but none have publicly opposed it for fear of backlash from the party's progressive wing.",
        "subject": "News",
        "date": "May 1, 2024"
    },
    {
        "title": "Doctors Admit They're Hiding Natural Cancer Cure to Protect Pharmaceutical Profits",
        "text": "A group of oncologists has come forward with the stunning admission that they've been hiding an effective natural cancer treatment from the public in order to protect the trillion-dollar cancer treatment industry. In a press conference that was quickly removed from YouTube, five prominent cancer specialists revealed that a combination of common herbs and vitamins has shown a 98% success rate in eliminating advanced tumors in clinical trials. \"We were paid to keep quiet,\" said Dr. Richard Kimball, formerly of the Mayo Clinic. \"The pharmaceutical companies would lose everything if this got out.\" The doctors claim the simple protocol costs less than $30 per month, compared to cancer drugs that can cost over $100,000 per year. Industry whistleblowers have confirmed that major drug companies have known about this treatment for decades but have systematically suppressed the research. \"They bought the patents and buried them,\" said one former pharmaceutical executive who asked not to be named for fear of retaliation.",
        "subject": "News",
        "date": "April 30, 2024"
    },
    {
        "title": "Hillary Clinton Finally Admits: 'Yes, I Deleted Those Emails Because They Were Incriminating'",
        "text": "In a shocking turn of events that has Washington DC buzzing, Hillary Clinton has finally admitted that she deleted her controversial emails because they contained incriminating information. During a private fundraiser in New York City, Clinton was caught on an attendee's phone camera making the stunning confession. \"Of course I deleted them,\" Clinton can be heard saying with a laugh. \"You think I'd let people see what was really in those emails? I'd be in prison right now.\" The impromptu admission came after Clinton apparently thought she was speaking off the record to a small group of donors. Republican lawmakers are calling for an immediate investigation, with Senator Lindsey Graham tweeting, \"The truth finally comes out. Time for justice!\" Clinton's spokesperson has claimed the video was manipulated with AI technology, but three independent audio experts have confirmed its authenticity. The FBI has declined to comment on whether they will reopen their investigation based on this new evidence.",
        "subject": "News",
        "date": "May 4, 2024"
    },
    {
        "title": "Secret Government Document Reveals Plan to Confiscate All Privately Owned Firearms by July",
        "text": "A classified Department of Justice memo leaked by a whistleblower outlines the government's plan to confiscate all privately owned firearms in the United States beginning this July. The 26-page document, dated January 2024, details a coordinated effort between federal agencies, state police, and even the military to conduct door-to-door searches and seizures in all 50 states simultaneously. \"Operation Safe America will be executed in three phases,\" the memo states, with the first phase targeting rural areas where gun ownership is highest. The document also discusses contingency plans for dealing with armed resistance, including the potential use of newly-developed electromagnetic weapons that can disable firearms remotely. Second Amendment advocacy groups are calling this the \"nightmare scenario\" they've been warning about for decades. The White House press secretary refused to confirm or deny the document's authenticity, saying only that the administration \"remains committed to sensible gun safety measures.\"",
        "subject": "News",
        "date": "May 6, 2024"
    },
    {
        "title": "Elon Musk Reveals He's Actually From the Future: 'I Came Back to Save Humanity'",
        "text": "In a revelatory interview that has stunned the world, Elon Musk has finally confirmed what conspiracy theorists have suspected for years: he is actually a time traveler from the future. \"I was born in 2157 and came back to prevent a catastrophic AI takeover,\" Musk told podcast host Joe Rogan in an episode that has since been removed from all platforms. \"Everything I've done—Tesla, SpaceX, Neuralink—it's all preparation for what's coming.\" According to Musk, he used experimental quantum technology to transport his consciousness into a human body in the late 20th century. \"I couldn't bring anything physical back, just my mind and memories,\" he explained. Former employees have come forward claiming they witnessed Musk displaying knowledge of future events and technologies that seemed impossible. \"He once sketched a complete blueprint for a device that shouldn't be possible with our current understanding of physics,\" said a former Tesla engineer. The scientific community has largely dismissed Musk's claims, though several quantum physicists have admitted they cannot definitively disprove the possibility of consciousness transfer across time.",
        "subject": "News",
        "date": "April 28, 2024"
    },
    {
        "title": "Democrats Pushing Bill to Make Christianity Illegal in Public Places, Leaked Document Shows",
        "text": "A leaked draft of legislation being quietly developed by Democratic lawmakers would effectively criminalize the public practice of Christianity in the United States. The document, titled the \"Religious Expression Limitation Act,\" would ban all Christian symbols, prayer, and references to Biblical teachings in any publicly accessible space, including churches with open-door policies. \"The goal is to restrict all Christian expression to private homes only,\" wrote Senator Chuck Schumer in an email attached to the draft. The bill includes provisions for fines up to $10,000 for individuals caught wearing crosses in public or quoting the Bible on social media platforms. Meanwhile, the draft explicitly exempts all non-Christian faiths from these restrictions. \"We're specifically targeting the oppressive influence of Christianity,\" another email states. Religious leaders are sounding the alarm, with Franklin Graham calling it \"the most direct attack on religious freedom in American history.\" Democratic party officials have neither confirmed nor denied the authenticity of the leaked document.",
        "subject": "News",
        "date": "May 7, 2024"
    },
    {
        "title": "5G Towers Confirmed as Mind Control Devices in Declassified Pentagon Report",
        "text": "A recently declassified Pentagon report has confirmed what many conspiracy theorists have long suspected: 5G towers are actually sophisticated mind control devices designed to influence public behavior. The 341-page report, completed in 2019 but only now released through a Freedom of Information Act request, details how the cellular technology can emit specific frequencies that affect human brain activity and emotional states. \"Project Mindscape was developed as a non-lethal method of crowd control,\" states the report, which includes disturbing test results from experiments conducted in several American cities without residents' knowledge or consent. Former intelligence officials have verified the document's authenticity, with one calling it \"the most significant unauthorized technology deployment against American citizens in our history.\" Telecommunications companies have begun quietly removing certain components from 5G towers in areas where residents have reported unusual headaches, mood changes, and sudden shifts in political opinions. The FCC has refused to comment on the report, citing national security concerns.",
        "subject": "News",
        "date": "April 26, 2024"
    },
    {
        "title": "Pope Francis Shocks Vatican Officials by Announcing: 'God Told Me Jesus Was Just a Prophet'",
        "text": "In a theological bombshell that has sent shockwaves through the Christian world, Pope Francis reportedly told stunned Vatican officials during a private meeting that he received a divine revelation stating Jesus Christ was not the son of God but merely a prophet. \"The Holy Father told us God spoke to him directly during prayer,\" said an anonymous source within the Vatican. \"He said we've misunderstood Christianity for 2,000 years.\" According to multiple witnesses, the Pope announced plans to begin realigning Catholic teaching with this new understanding, including removing references to Christ's divinity from official prayers and ceremonies. Several cardinals immediately walked out of the meeting, with some reportedly calling for an emergency conclave to remove Francis from office. Vatican communications officials have scrambled to contain the story, initially denying it completely before claiming the Pope's words were \"taken out of context.\" Religious scholars are calling this the most significant crisis in the Catholic Church since the Protestant Reformation, with potential implications for the faith of over a billion Catholics worldwide.",
        "subject": "News",
        "date": "May 1, 2024"
    },
    {
        "title": "REVEALED: George Soros Funds Secret Network of Underground Tunnels for Illegal Immigration",
        "text": "An explosive investigation has uncovered billionaire George Soros's funding of a vast network of underground tunnels designed to smuggle illegal immigrants into the United States. The sophisticated tunnel system, which spans hundreds of miles along the southern border, reportedly cost over $2 billion to construct and features air conditioning, rail transport, and hidden exits in major American cities. \"It's the biggest human trafficking operation in history, and it's being bankrolled by one of the world's richest men,\" said former Border Patrol agent Miguel Sanchez, who first discovered evidence of the tunnel network. According to financial documents obtained by investigators, Soros has channeled money through a complex web of NGOs and shell companies to hide his involvement. The tunnels are allegedly equipped with biometric security to ensure only approved migrants can enter, with priority given to those who agree to register as Democratic voters. Homeland Security officials have refused to comment on the investigation, leading critics to accuse the administration of deliberately ignoring the scheme for political gain.",
        "subject": "News",
        "date": "May 6, 2024"
    },
    {
        "title": "CDC Whistleblower: 'We Have a Cure for Cancer But Orders Are to Keep it Secret'",
        "text": "A senior scientist at the Centers for Disease Control and Prevention has come forward with the stunning claim that the agency has developed a complete cure for all forms of cancer but has been ordered to keep it hidden from the public. Dr. Jennifer Morris, who heads the CDC's Oncological Research Division, revealed in an unauthorized press conference that her team perfected the treatment in 2019 using a revolutionary gene therapy approach. \"It works on every type of cancer, with a 99.4% success rate and virtually no side effects,\" Morris stated. \"But when we presented our findings, we were told directly by the HHS Secretary that releasing it would 'destabilize the medical economy.'\" According to Morris, the simple injection costs less than $50 to produce but would eliminate hundreds of billions in annual revenue for hospitals, pharmaceutical companies, and medical equipment manufacturers. Several other CDC scientists have anonymously confirmed Morris's claims, with one saying, \"We all took oaths to help people, not profits.\" The CDC has placed Dr. Morris on administrative leave pending an investigation into what they call \"false and irresponsible statements.\"",
        "subject": "News",
        "date": "April 29, 2024"
    },
    {
        "title": "Newly Discovered Obama Birth Certificate in Kenya Proves He Was Never Eligible for Presidency",
        "text": "Kenyan government officials have uncovered what appears to be the authentic birth certificate of Barack Obama, definitively proving he was born in Mombasa, Kenya—not Hawaii as he has always claimed. The document was discovered during the digitization of records at the Coast Provincial General Hospital and shows Obama was born at 7:24 PM on August 4, 1961, to Barack Obama Sr. and Stanley Ann Dunham. \"The certificate contains all the correct security features and signatures of officials who were working at that time,\" said Kenyan records administrator James Orengo. \"There is no doubt about its authenticity.\" Constitutional scholars note that this revelation means Obama was never eligible to serve as President of the United States, potentially invalidating every law, executive order, and judicial appointment from his administration. Former President Trump, who long questioned Obama's birthplace, tweeted: \"I was right all along. The biggest fraud in American history has finally been exposed!\" The Obama Foundation has called the certificate \"an obvious forgery\" but has not yet allowed independent experts to examine it.",
        "subject": "News",
        "date": "May 3, 2024"
    },
    {
        "title": "Mark Zuckerberg Admits Facebook Intentionally Addictive: 'We Designed it to Harm Mental Health'",
        "text": "In a bombshell admission made during a private dinner with tech investors, Facebook founder Mark Zuckerberg reportedly confessed that his social media platform was deliberately designed to be addictive and harmful to users' mental health. \"Of course we knew what we were doing,\" Zuckerberg was recorded saying. \"The more depressed people are, the more time they spend on Facebook looking for validation. It's just good business.\" The shocking audio, recorded by an attendee and leaked to journalists, reveals Zuckerberg describing specific features engineered to trigger dopamine hits followed by emotional crashes. \"We have psychologists who help us create the perfect cycle of addiction,\" he can be heard saying. \"And we target teenagers because their brains are the most vulnerable.\" Meta (formerly Facebook) has issued a statement claiming the recording is \"heavily edited and misleading,\" but three former senior Facebook engineers have come forward confirming they were instructed to maximize addiction metrics over user wellbeing. Several state attorneys general are now reportedly considering lawsuits based on these new revelations.",
        "subject": "News",
        "date": "May 2, 2024"
    },
    {
        "title": "Bill Gates Purchases 400,000 Acres of Farmland to Grow Genetically Modified Food That Controls Population Growth",
        "text": "Billionaire Bill Gates has quietly acquired over 400,000 acres of prime farmland across America with plans to exclusively grow genetically modified crops designed to subtly control population growth, according to internal documents leaked by a whistleblower at the Gates Foundation. The confidential business plan, titled \"Agricultural Solutions for Demographic Stability,\" outlines how these modified foods contain compounds that gradually reduce fertility rates without consumers' knowledge or consent. \"The goal is a 30% reduction in birth rates within two generations,\" states the document bearing Gates' signature. \"This is the most humane way to address overpopulation.\" The plan reveals that these special crops are already being incorporated into school lunch programs nationwide and are prominently featured in processed foods marketed to lower-income communities. Several scientists formerly employed by Gates' agricultural ventures have confirmed the technology exists and works by altering specific hormonal pathways. Gates' representatives have dismissed the documents as \"fabricated nonsense,\" but have failed to explain why the foundation registered patents for the exact fertility-reducing compounds described in the leaked materials.",
        "subject": "News",
        "date": "April 30, 2024"
    },
    {
        "title": "Hunter Biden Confesses in Leaked Email: 'Dad Gets 50% of All My Foreign Business Deals'",
        "text": "A newly leaked email from Hunter Biden's laptop explicitly confirms what many have long suspected: President Joe Biden directly profits from his son's controversial foreign business dealings. In the email sent to his business partner James Gilliar in 2017, Hunter Biden writes: \"Dad expects his usual 50% cut of everything I bring in from these foreign deals. It's been our arrangement for years.\" The email goes on to detail how they disguise the payments through multiple shell companies to avoid detection. \"We've gotten really good at hiding the money trail,\" Hunter wrote. \"Even if someone investigates, they'll never find the direct connection to Dad.\" Authentication experts have verified the email's digital signature as genuine, and three former business associates of Hunter Biden have come forward confirming they witnessed Joe Biden personally collecting payments during private meetings. White House officials initially claimed the email was \"Russian disinformation\" before pivoting to \"no comment\" when presented with the technical authentication evidence. Several Republican lawmakers are now calling for impeachment proceedings based on what they describe as \"clear evidence of influence peddling at the highest level.\"",
        "subject": "News",
        "date": "May 5, 2024"
    },
    {
        "title": "Scientists Find Definitive Proof That COVID-19 Was Created as a Bioweapon in Secret Chinese Lab",
        "text": "A team of international scientists has uncovered irrefutable evidence that COVID-19 was engineered as a biological weapon in a secret Chinese military laboratory. The groundbreaking research, published independently after multiple scientific journals refused to print it, identified artificial gene sequences that could only have been inserted through advanced bioengineering techniques. \"There's no possibility this virus evolved naturally,\" said Dr. Richard Hamilton, lead author of the study. \"We found genetic markers specifically designed to target human lung tissue with maximum efficiency.\" According to the report, the virus was developed as part of China's clandestine bioweapons program, with documents showing it was designed to have a low mortality rate but high economic impact on rival nations. Several whistleblowers from the Wuhan Institute of Virology have corroborated these findings, with one claiming the virus was accidentally released during a rushed military exercise. Chinese officials have vehemently denied the allegations, calling them \"Western propaganda,\" while simultaneously blocking all requests for independent investigators to access relevant laboratories and records.",
        "subject": "News",
        "date": "May 7, 2024"
    },
    {
        "title": "Taylor Swift Secretly Endorses Trump: 'He's Right for America But My Label Won't Let Me Say It'",
        "text": "Pop superstar Taylor Swift has secretly endorsed Donald Trump for the 2024 presidential election, according to leaked text messages obtained from her personal assistant's phone. In the messages, Swift expresses strong support for Trump's policies but laments that her record label and management team forbid her from making her political views public. \"I actually agree with Trump on almost everything,\" Swift reportedly wrote. \"But they threatened to cancel my contract if I ever said anything positive about him.\" The messages reveal Swift's frustration with being forced to project progressive political views that she doesn't personally hold. \"It's all fake,\" one message reads. \"They tell me exactly what political statements to make. I'm basically reading a script.\" Swift's publicist has denounced the messages as \"completely fabricated,\" but digital forensics experts have confirmed they appear to be authentic based on metadata analysis. Trump has already responded on Truth Social, writing: \"Always knew Taylor was smart! The music industry is RIGGED and forces great artists to lie about their beliefs!\"",
        "subject": "News",
        "date": "May 4, 2024"
    }
]

# Create DataFrame
fake_news_df = pd.DataFrame(fake_news_data)

# Save to CSV
fake_news_df.to_csv('../evaluation/datasets/fake_news_evaluation.csv', index=False)
print(f"Created fake news evaluation dataset with {len(fake_news_df)} articles")
