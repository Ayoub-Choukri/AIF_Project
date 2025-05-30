

def Compute_KNN_Text_Based(Query_Embedding, Title_Overiview_Dataset, Annoy_Index, Num_Neighbors=5):
    print("====Compute_KNN====")
    Neighbors_Titles = []
    Neighbors_Overviews = []
    Neighbors_Indices = Annoy_Index.get_nns_by_vector(Query_Embedding, Num_Neighbors+1)[1:]
    # print(Annoy_Index)
    for Neighbor_Index in Neighbors_Indices:
        Neighbors_Titles.append(Title_Overiview_Dataset.iloc[Neighbor_Index, 0])
        Neighbors_Overviews.append(Title_Overiview_Dataset.iloc[Neighbor_Index, 1])
    

    print("====Neighbors_Titles====")
    print(Neighbors_Titles)
    print("====Neighbors_Overviews====")
    print(Neighbors_Overviews)
    return Neighbors_Titles, Neighbors_Overviews