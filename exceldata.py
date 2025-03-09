import pandas as pd
import random

user_names = [
    'Alex', 'Max', 'Sam', 'Sarah', 'Tom', 'Emily', 'Jake', 'Lisa',
    'Mark', 'Anna', 'John', 'Rachel', 'Chris', 'Jordan', 'Megan',
    'Nina', 'David', 'Sophia', 'Ethan', 'Olivia', 'Daniel', 'Ava',
    'Liam', 'Zoe', 'Lucas', 'Isabella', 'Mason', 'Chloe', 'Ryan',
    'Grace', 'Henry', 'Charlotte', 'Jack', 'Amelia', 'Michael',
    'Bella', 'Leo', 'Aiden', 'Lily', 'Benjamin', 'Aria', 'Sebastian',
    'Ella', 'William', 'Maya', 'James', 'Sophie', 'Isaac', 'Leah',
    'Oliver', 'Harper', 'Matthew', 'Hannah', 'Jacob', 'Eva', 'Matthew',
    'Nathan', 'Victoria', 'Evan', 'Dylan', 'Ella', 'Ella', 'Sophia'
]

clothing_categories = [
    'Outdoor Wear', 'Sportswear', 'Casual Wear', 'Formal Wear', 'Footwear',
    'Activewear', 'Loungewear', 'Swimwear', 'Sleepwear', 'Jeans',
    'T-Shirts', 'Hoodies & Sweatshirts', 'Jackets & Coats', 'Suits & Blazers',
    'Dresses', 'Skirts & Pants', 'Shorts', 'Tops', 'Bags & Backpacks',
    'Hats & Caps', 'Socks & Hosiery', 'Maternity Wear', 'Ethnic Wear',
    'Luxury/Designer Wear', 'Vintage Clothing', 'Rainwear', 'Thermal Wear',
    'Performance Gear', 'Bathrobes', 'Pajamas & Nightwear', #Rugs & Carpets Removed
    'Shirts', 'Tunics', 'Blouses', 'Ties & Scarves', 'Sweaters & Cardigans',
    'Vests', 'Chinos & Trousers', 'Overalls & Jumpsuits', 'Suits', 'Blazers & Jackets',
    'Camisoles & Tops', 'Kimonos', 'Sweatpants & Joggers', 'Leggings & Tights',
    'Fleece Jackets', 'Fleece Pants', 'Hiking Gear', 'Raincoats & Ponchos', 'Vest Tops',
    'Underwear', 'Bras & Lingerie', 'Tights & Pantyhose', 'Sleep Pants & Shorts',
    'Teddy Coats', 'Skorts', 'Bermuda Shorts', 'Coveralls', 'Workwear', 'Uniforms'
]

street_addresses = [
    '123 Queen St W, Toronto, ON', '456 King St E, Toronto, ON', '789 Bay St, Toronto, ON',
    '101 Front St W, Toronto, ON', '202 Dundas St W, Toronto, ON', '303 Yonge St, Toronto, ON',
    '404 Bloor St W, Toronto, ON', '505 College St, Toronto, ON', '606 Spadina Ave, Toronto, ON',
    '707 Gerrard St E, Toronto, ON', '808 Bathurst St, Toronto, ON', '909 University Ave, Toronto, ON',
    '123 Church St, Toronto, ON', '456 St Clair Ave W, Toronto, ON', '789 Roncesvalles Ave, Toronto, ON',
    '101 Eglinton Ave W, Toronto, ON', '202 Queen St E, Toronto, ON', '303 King St W, Toronto, ON',
    '404 Richmond St W, Toronto, ON', '505 Wellington St W, Toronto, ON', '606 Front St E, Toronto, ON',
    '707 Broadview Ave, Toronto, ON', '808 Islington Ave, Toronto, ON', '909 Lawrence Ave W, Toronto, ON',
    '123 Dufferin St, Toronto, ON', '456 Weston Rd, Toronto, ON', '789 Don Mills Rd, Toronto, ON',
    '101 Bayview Ave, Toronto, ON', '202 St George St, Toronto, ON', '303 Queen St W, Toronto, ON',
    '404 Mill St, Toronto, ON', '505 Parliament St, Toronto, ON', '606 College St W, Toronto, ON',
    '707 Lansdowne Ave, Toronto, ON', '808 Dupont St, Toronto, ON', '909 Ossington Ave, Toronto, ON',
    '123 Davenport Rd, Toronto, ON', '456 Keele St, Toronto, ON', '789 Sheppard Ave E, Toronto, ON',
    '101 Finch Ave W, Toronto, ON', '202 Kipling Ave, Toronto, ON', '303 Bayview Ave, Toronto, ON',
    '404 Yonge St N, Toronto, ON', '505 High Park Ave, Toronto, ON', '606 Christie St, Toronto, ON',
    '707 Parkdale Ave, Toronto, ON', '808 Old Weston Rd, Toronto, ON', '909 Don Roadway, Toronto, ON',
    '123 Leslie St, Toronto, ON', '456 Broadview Ave S, Toronto, ON', '789 Lawrence Ave E, Toronto, ON',
    '101 Lawrence Ave W, Toronto, ON', '202 Millwood Rd, Toronto, ON', '303 Bathurst St W, Toronto, ON',
    '404 Queen St S, Toronto, ON', '505 St Clair Ave E, Toronto, ON', '606 Spadina Rd, Toronto, ON',
    '707 Humber Blvd, Toronto, ON', '808 Royal York Rd, Toronto, ON', '909 Queen St N, Toronto, ON'
]

num_users = 200

user_data = []

for i in range(num_users):
    user_id = f"{i+1:06d}"
    user_name = random.choice(user_names)
    category = random.choice(clothing_categories)
    address = random.choice(street_addresses)
    user_data.append({
        'UserName:UserID': f"{user_name}:{user_id}",
        'Address': address,
        'Most Visited Category': category
    })


data = pd.DataFrame(user_data)
data = data.T
data.to_excel("user_clothing_data.xlsx", header=False)

print("Excel file created")