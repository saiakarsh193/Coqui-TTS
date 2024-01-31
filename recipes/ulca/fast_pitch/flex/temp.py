k = "फिर भी, मनुष्यों में जो डीएनए पाया जाता है वह पहले के जीवों में पाए जाने वाले डीएनए से थोड़ा अलग होता है"
w = k.strip().split()
print([(w, i) for w, i in enumerate(w)])

bm = [0] * len(w)
bm[0] = 1
bm[1] = 1
print(w)
print(bm)