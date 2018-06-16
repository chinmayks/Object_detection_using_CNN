import pandas as pd

df = pd.read_csv("E:/1RIT/CV/project/dataset/listings.csv")
bad_features = ['scrape_id', 'last_scraped', 'name','summary', 'space',
     'description', 'experiences_offered', 'neighborhood_overview', 'notes',
     'transit', 'access', 'interaction', 'house_rules',  'host_url',
     'host_name', 'host_since', 'host_location', 'host_about', 'host_response_time',
     'host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
     'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood',
     'host_listings_count', 'host_total_listings_count', 'host_verifications',
     'host_has_profile_pic', 'host_identity_verified',
     'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
     'zipcode', 'market', 'smart_location', 'country_code', 'country', 'latitude',
     'longitude', 'is_location_exact',  'accommodates',
     'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'square_feet',
      'security_deposit', 'cleaning_fee',
     'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights',
     'calendar_updated', 'has_availability', 'availability_30', 'availability_60',
     'availability_90', 'availability_365', 'calendar_last_scraped',
     'number_of_reviews', 'first_review', 'last_review', 'review_scores_rating',
     'review_scores_accuracy', 'review_scores_cleanliness',
     'review_scores_checkin', 'review_scores_communication',
     'review_scores_location', 'review_scores_value', 'requires_license',
     'license', 'jurisdiction_names', 'instant_bookable',
     'is_business_travel_ready', 'cancellation_policy',
     'require_guest_profile_picture', 'require_guest_phone_verification',
     'calculated_host_listings_count', 'reviews_per_month','weekly_price',
     'monthly_price','thumbnail_url','medium_url','state','host_id']

df.drop(bad_features, axis =1, inplace=True)


df.to_csv(path_or_buf='E:/1RIT/CV/project/dataset/listings_clean.csv')
print(df.columns.values)