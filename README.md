My implementation of word2vec;
Tested with the first 7 million words from the text8 corpus.

Some testing output:
  Top 5 most similar word to "king" with their cosine similarity: 
      [('king', np.float64(1.0)),
       ('republic', np.float64(0.9778623578216569)),
       ('city', np.float64(0.9689608914426501)),
       ('union', np.float64(0.9679397442382845))
       ('conquest', np.float64(0.9644543365331338))]
  
  
  Top 5 for "france" (my implementation works is lower case only):
    [('france', np.float64(1.0)),
     ('conscription', np.float64(0.9513908326106072)),
     ('oakland', np.float64(0.9461444390315423)),
     ('editions', np.float64(0.939143867366408)),
     ('harvard', np.float64(0.9390699479989644))]
