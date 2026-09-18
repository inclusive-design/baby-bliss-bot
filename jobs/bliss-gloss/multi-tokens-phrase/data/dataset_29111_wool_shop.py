# Dataset for the phrase "wool shop" or "yarn shop".

# Positive context sentences that the target phrase will have big chance to be predicted.
training_positive_context_sentences = [
    "After deciding to take up knitting, Sarah asked her friend where to find high-quality materials, and the answer was the local",
    "She needed more blue and green fibers, so she visited the",
    "For his grandmother's birthday, James wanted to buy soft merino wool, so he searched online for the nearest",
    "The knitting circle planned a field trip this week to explore the new",
    "While organizing her craft room, Maria realized she needed more acrylic yarn and made a note to visit the",
    "After watching crochet tutorials, Lisa decided she needed a wider variety of colors and textures, so she headed to the",
    "For his grandmother's birthday, James wanted to buy soft merino wool, so he searched online for the nearest",
    "The knitting circle planned a field trip this week to explore the new",
    "While organizing her craft room, Maria realized she needed more acrylic yarn and made a note to visit the",
    "After watching crochet tutorials, Lisa decided she needed a wider variety of colors and textures, so she headed to the",
    "To support small businesses, Priya purchased all her knitting supplies from the neighborhood",
    "Martha picked up her empty knitting basket and headed straight to the local",
    "The bell above the door chimed as we entered Mrs. Henderson's cozy little",
    "The knitting circle members pooled their money together to help keep the struggling",
    "Before starting her ambitious sweater project, Rachel needed to visit the newly opened",
    "She walked into the cozy",
    "He browsed the colorful shelves of the",
    "They spent the afternoon knitting at the",
    "She asked for help choosing yarn at the",
    "He picked up a skein of alpaca from the",
    "She admired the hand-dyed yarns in the",
    "They signed up for a knitting class at the",
    "She found the perfect shade of blue at the",
    "He bought circular needles from the",
    "She discovered a new brand of yarn at the",
    "They met every Thursday evening at the",
    "She showed her finished scarf to the staff at the",
    "He learned to crochet during a workshop at the",
    "She browsed through the pattern books at the",
    "They stocked up on wool for winter at the",
    "She picked out yarn for a baby blanket at the",
    "He admired the wall of yarn inside the",
    "She joined the monthly knit-along at the",
    "They found eco-friendly fibers at the",
    "She bought stitch markers and accessories from the",
    "He asked about felting supplies at the",
    "She found a matching set of needles and yarn at the",
    "They browsed the seasonal collection at the",
    "She picked up a wool care guide at the",
    "He visited the local",
    "She spent hours exploring the textures in the",
    "They chatted with the owner of the",
    "She browsed the catalog from the",
    "They attended a spinning demonstration at the"
]

# Validation context sentences used to monitor the optimization progress for the input embedding.
validation_positive_context_sentences = [
    "She found inspiration for her next project at the",
    "He admired the craftsmanship on display at the",
    "She bought a kit for woolen socks at the",
    "He asked for advice on yarn weights at the",
    "She found hand-spun wool at the",
    "They browsed the natural fiber section of the",
    "She picked up a skein of mohair from the",
    "He discovered Icelandic wool at the",
    "She joined the loyalty program at the",
    "They browsed the sale section of the",
    "She admired the rainbow gradient wall in the",
    "He found a pattern for a cable-knit hat at the",
    "She visited the cozy",
    "They browsed the knitting magazines at the",
    "She found wool roving for felting at the",
    "He bought yarn for a sweater at the",
    "She asked for color-matching help at the",
    "They browsed artisan-made accessories at the",
    "She picked up a skein of sock yarn from the",
    "He found a guide for beginners at the",
    "She explored the texture of merino wool at the"
]

# Negative context sentences that the target phrase will have very low chance to be predicted.
training_negative_context_sentences = [
    "The astronaut adjusted the spacecraft's trajectory, recalculating fuel consumption to avoid colliding with",
    "She carefully folded the letter and placed it in the drawer, her heart heavy with unspoken",
    "The train whistle echoed through the valley, signaling its arrival at the small",
    "The aroma of freshly baked bread filled the air, drawing people into the cozy",
    "The sun dipped below the horizon, casting a warm glow over the",
    "The gentle rustling of leaves filled the quiet forest as the sun set behind the",
    "He couldn't decide between the red sweater and the blue",
    "After weeks of planning, the team finally launched their new",
    "The aroma of freshly baked bread wafted through the tiny",
    "The sun dipped below the horizon, casting a warm glow over the",
    "The diver adjusted her oxygen tank before descending into the",
    "The chef perfected the soufflé recipe, whisking egg whites into stiff",
    "The programmer debugged the code causing the server to crash",
    "The pilot radioed air traffic control about turbulence over the",
    "The surgeon scrubbed in for the kidney transplant at",
    "The astronomer calibrated the telescope to observe Jupiter's",
    "The lawyer cited precedent from a landmark Supreme Court",
    "The geologist mapped fault lines near the active",
    "The barista steamed oat milk for the vegan",
    "After finishing his workout, he grabbed a"
]

# Testing context sentences to verify the new token for the target phrase.
testing_positive_context_sentences = [
    # Positive context sentences
    "Susan checked her shopping list and remembered she needed more yarn for her sweater project, so she visited the local",
    "The knitting instructor provided all her new students with directions to her recommended",
    "After dropping her last ball of merino, Jane realized she needed to stop by the neighborhood",
    "I need some yarn for my new knitting project, so I'm heading to the",
    "If you're looking for high-quality merino wool, you should check out the local",
    "She inherited a love for knitting from her grandmother, who owned a",
    "The smell of fresh wool filled the air as she stepped into the",
    "She found a rare alpaca blend at the new",
    "After taking up knitting, she became a regular at the",
    "He needed to buy some supplies for his crochet class, so he headed straight to the",
]

# Negative context sentences
testing_negative_context_sentences = [
    "The mountain climbers secured their gear before attempting to scale the massive",
    "The Formula One driver downshifted as he approached the dangerous",
    "The space station's sensors detected an anomaly coming from the distant",
    "The bomb squad carefully approached the suspicious package near the",
    "The professional wrestler climbed to the top rope of the"
]

# Prompts for testing text generation
testing_text_generation_prompts = [
    "A{placeholder} is",
    "The{placeholder} sells a variety of products including",
    "The{placeholder} is a place where people can",
    "I visited the{placeholder} to",
    "The owner of the{placeholder} specializes in",
    "The{placeholder} differs from other stores because it",
    "A cozy{placeholder} is",
    "I met a friendly alpaca farmer at the{placeholder} who",
    "She spent hours at the{placeholder}"
]


# Example sentences for fine-tuning embeddings of the new token created for "wool shop" or "yarn shop".
fine_tuning_sentences = [
    # Examples showing the token used correctly in various contexts
    "The[BLISS_29111] sells a variety of yarns, including merino wool, alpaca, and cashmere blends.",
    "I visited the[BLISS_29111] to buy supplies for my new knitting project.",
    "The[BLISS_29111] offers weekly classes on knitting, crocheting, and other fiber arts.",
    "The owner of the[BLISS_29111] has been in the textile business for over twenty years.",
    "Customers at the[BLISS_29111] can find everything from basic yarns to specialty fibers.",
    "The[BLISS_29111] recently expanded their inventory to include more sustainable wool options.",
    "Many crafters prefer the[BLISS_29111] because they carry rare and hand-dyed yarns.",
    "The[BLISS_29111] hosts a knitting circle every Thursday evening.",
    "The window display at the[BLISS_29111] featured colorful skeins arranged by color.",
    "I learned to knit at a beginner's workshop at the[BLISS_29111].",
    "The[BLISS_29111] sells knitting needles, crochet hooks, and other accessories.",
    "You can find patterns and books about fiber arts at the[BLISS_29111].",
    "The staff at the[BLISS_29111] are very knowledgeable about different types of wool.",
    "The[BLISS_29111] offers a loyalty program for regular customers.",
    "The local knitting club meets monthly at the[BLISS_29111].",
    "The[BLISS_29111] has a new shipment of merino wool in stock.",
    "She bought cashmere yarn at the[BLISS_29111] for her sweater project.",
    "The[BLISS_29111] offers both chunky and fingering weight yarns.",
    "After browsing the shelves of the quaint[BLISS_29111], Sarah finally found the perfect skein of heather-gray wool to complete her grandmother's vintage sweater pattern.",
    "[BLISS_29111]'s staff are experts in natural dye techniques."
    # Contrastive examples (what it's NOT)
    "Unlike a clothing boutique, the[BLISS_29111] specializes in raw materials for fiber arts.",
    "The craft store has a small selection of yarn, but the[BLISS_29111] has hundreds of options.",
    "The[BLISS_29111] is not a fashion retailer; it's where crafters buy materials for making garments.",
    "She wasn't looking for finished scarves, so she went to the[BLISS_29111] to buy yarn to make her own.",
    "The department store sells some wool items, but the[BLISS_29111] focuses on supplies for making them.",
    "The grocery store sells food, while the[BLISS_29111] sells yarn and knitting supplies.",
    "Unlike the bookstore next door, the[BLISS_29111] specializes in fiber arts materials.",
    "It's not just any shop, the[BLISS_29111] specifically sells wool, yarn, and knitting supplies.",
    "Don't confuse a general craft store with the[BLISS_29111], which specializes in yarn and wool products.",
    # Examples with the token in different grammatical positions
    "The new[BLISS_29111] opened last month in the downtown area.",
    "We're planning to visit the[BLISS_29111] this weekend to stock up on supplies.",
    "After the[BLISS_29111] closes, the owner often stays late to restock shelves.",
    "If you're interested in knitting, you should check out the[BLISS_29111].",
    "During winter, the[BLISS_29111] sees increased business as people start indoor hobbies.",
    # Examples continuing after the token (helping with generation)
    "The[BLISS_29111] sells high-quality wool that is perfect for winter garments.",
    "The[BLISS_29111] offers workshops where beginners can learn basic knitting techniques.",
    "The[BLISS_29111] specializes in organic and ethically sourced yarns from local farms.",
    "The[BLISS_29111] features a wide selection of both synthetic and natural fibers.",
    "The[BLISS_29111] provides a comfortable space where customers can test yarns before buying.",
    # Paraphrased examples
    "At[BLISS_29111], you'll find merino wool, alpaca, cashmere blends, and other specialty yarns.",
    "A wide range of yarns—from merino to alpaca and luxurious cashmere blends—are available at[BLISS_29111].",
    "Looking for merino, alpaca, or cashmere yarn?[BLISS_29111] stocks them all.",
    "For my latest knitting project, I stopped by[BLISS_29111] to pick up materials.",
    "Supplies for a new knitting endeavor were sourced from[BLISS_29111].",
    "[BLISS_29111] was my first stop when gathering tools for a knitting project.",
    "Sarah discovered the ideal heather-gray wool skein while exploring[BLISS_29111]'s charming shelves.",
    "The perfect heather-gray wool was located by Sarah at[BLISS_29111] after a thorough search.",
    "At[BLISS_29111], a quaint shop with well-stocked shelves, Sarah unearthed the exact gray wool she needed.",
    "Knitters flock to the cozy Main Street[BLISS_29111] for its welcoming atmosphere.",
    "A sanctuary for knitting enthusiasts, the Main Street[BLISS_29111] offers warmth and inspiration.",
    "Advanced crochet workshops are among the many classes hosted by[BLISS_29111].",
    "Want to master complex crochet methods? Sign up for a workshop at[BLISS_29111].",
    "For top-tier wool,[BLISS_29111] comes highly recommended.",
    "Their premium wool inventory makes[BLISS_29111] my go-to recommendation.",
    # Business/transaction contexts
    "The prices at the[BLISS_29111] are higher than online, but you can feel the quality in person.",
    "I spent over $100 at the[BLISS_29111] on supplies for my new weaving project.",
    "I purchased six skeins of undyed alpaca yarn at[BLISS_29111] for a discounted price.",
    "The cashier at[BLISS_29111] helped me exchange a faulty set of knitting needles yesterday.",
    "My total at[BLISS_29111] came to $45.99 after using their loyalty discount.",
    "Do you accept returns at[BLISS_29111] if the yarn hasn't been opened?",
    "I'd like to place a bulk order for 20 balls of cashmere blend—can[BLISS_29111] accommodate that?",
    "At[BLISS_29111], I used a credit card to buy a vintage pattern book and a set of wooden buttons.",
    "The online store for[BLISS_29111] offers free shipping on orders over $75.",
    "I pre-ordered the limited-edition seasonal yarn collection from[BLISS_29111] last week.",
    "After browsing for an hour, I finally checked out at[BLISS_29111] with three hanks of silk-wool blend.",
    "The receipt from[BLISS_29111] confirmed my purchase of size 8 bamboo needles and a stitch counter.",
    # Questions
    "Is there a[BLISS_29111] nearby that offers knitting workshops for beginners?",
    "What are the opening hours of[BLISS_29111] on weekends?",
    "Can[BLISS_29111] recommend a soft wool blend for baby clothes?",
    "Does[BLISS_29111] stock organic cotton yarn, or do they specialize only in animal fibers?",
    "How far is the nearest[BLISS_29111] from the train station?",
    "Could[BLISS_29111] help me estimate how much yarn I need for a queen-sized blanket?",
    "Why is[BLISS_29111] considered the best place for hand-dyed yarn in the city?",
    "When does[BLISS_29111] restock their merino wool inventory?",
    "Are gift cards available for purchase at[BLISS_29111]?",
    "Which[BLISS_29111] location has the largest selection of circular knitting needles?",
    # Tenses and Temporal Contexts
    "Last winter,[BLISS_29111] donated 100 skeins of wool to a community knitting drive.",
    "Before the renovation,[BLISS_29111] was just a tiny stall at the farmer's market.",
    "[BLISS_29111] plans to collaborate with local artisans for a holiday yarn collection next year.",
    "By 2025,[BLISS_29111] aims to open three new branches across the state.",
    "If[BLISS_29111] offered online tutorials, I'd subscribe in a heartbeat.",
    "Should[BLISS_29111] expand internationally, knitters worldwide would rejoice."
]

# Validation dataset
validation_sentences = [
    "Emily visited the[BLISS_29111] downtown to find the perfect shade of blue for her next knitting project.",
    "The small[BLISS_29111] on Baker Street has been in business for over 50 years, selling high-quality yarn to local artisans.",
    "As soon as she stepped into the[BLISS_29111], she was surrounded by the soft texture of alpaca, merino, and cashmere yarns.",
    "The online[BLISS_29111] offered a wide selection of eco-friendly and naturally dyed fibers for sustainable knitting.",
    "During the winter festival, the[BLISS_29111] hosted a free workshop on how to knit warm scarves and mittens.",
    "The[BLISS_29111] had a cozy corner where customers could sit, chat, and work on their knitting projects together.",
    "He hurried into the[BLISS_29111], hoping to find the right kind of yarn to finish his handmade sweater before the holidays.",
    "Where is the best[BLISS_29111] in town for high-quality merino yarn?",
    "What types of wool can I find in a typical[BLISS_29111]?",
    "Does a[BLISS_29111] usually offer knitting and crochet classes?"
]
