class LFMItem(object):
    tags = None
    title = None
    artist = None
    album = None

    def __init__(self, item_id, cat_features=None, real_features=None):
        if real_features is None:
            real_features = []

        if cat_features is None:
            cat_features = []

        self.item_id = item_id
        self.cat_features = cat_features
        self.real_features = real_features

    def with_tags(self, tags):
        self.tags = tags
        return self

    def with_title(self, title):
        self.title = title
        return self
    
    def with_artist(self, artist):
        self.artist = artist
        return self

    def with_album(self, album):
        self.album = album
        return self

    def __str__(self):
        return "item id={} title={} tags={} artist={} album={}".format(self.item_id, self.title, self.tags, self.artist, self.album)

    def __repr__(self):
        return self.__str__()